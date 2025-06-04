from datasets import Dataset
from datasets import load_dataset_builder
from datasets import get_dataset_split_names
from datasets import load_dataset

from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import DataCollatorForLanguageModeling
from transformers import get_linear_schedule_with_warmup
import math

import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F

import wandb
import argparse
import json
import os
from tqdm import tqdm
import numpy as np

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

class BradleyTerryRewardModel(nn.Module):
    """Bradley-Terry reward model for preference learning"""
    def __init__(self, base_model_path):
        super().__init__()
        # Load the base model
        self.model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            torch_dtype="auto",
            device_map="auto"
        )
        # Add a reward head with the same dtype as the model
        self.reward_head = nn.Linear(self.model.config.hidden_size, 1)
        # Match the dtype of the base model
        self.reward_head = self.reward_head.to(self.model.dtype)
        
    def forward(self, input_ids, attention_mask):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
        # Use the last hidden state of the last token as the reward
        hidden_states = outputs.hidden_states[-1]
        # Get the last non-padding token for each sequence
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = input_ids.shape[0]
        last_hidden_states = hidden_states[range(batch_size), sequence_lengths]
        # Pass through reward head
        rewards = self.reward_head(last_hidden_states).squeeze(-1)
        return rewards

def main():
    args = parse_args()

    run = wandb.init(
        entity="ethanhellman",
        project="CS224R",
        config={
            "learning_rate": args.learning_rate,
            "model": args.sft_checkpoint,
            "dataset": args.dataset_name,
            "epochs": args.num_epochs,
            "batch_size": args.batch_size,
            "num_samples": args.num_samples,
            "temperature": args.temperature,
            "seed": args.seed
        }
    )

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"CUDA device count: {torch.cuda.device_count()}")

    tokenizer = AutoTokenizer.from_pretrained(args.sft_checkpoint)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load prompts for on-policy sampling
    prompt_dataloader = get_prompt_dataloader(args, "train")
    validation_dataloader = get_prompt_dataloader(args, "val")

    # Load policy model
    policy_model = AutoModelForCausalLM.from_pretrained(
        args.sft_checkpoint,
        torch_dtype="auto",
        device_map="auto"
    )
    policy_model.gradient_checkpointing_enable()

    # Load or train reward model
    if args.reward_model_path and os.path.exists(os.path.join(args.reward_model_path, "reward_model.pt")):
        print(f"Loading reward model from {args.reward_model_path}")
        # Initialize reward model with the base SFT checkpoint
        reward_model = BradleyTerryRewardModel(args.sft_checkpoint)
        # Load the saved state dict
        reward_model.load_state_dict(torch.load(os.path.join(args.reward_model_path, "reward_model.pt"), map_location=device))
    else:
        print("Training Bradley-Terry reward model...")
        reward_model = train_bradley_terry_reward(args, tokenizer, device, run)
        # Save reward model
        os.makedirs("./checkpoints/reward_model", exist_ok=True)
        torch.save(reward_model.state_dict(), "./checkpoints/reward_model/reward_model.pt")
    
    reward_model = reward_model.to(device)
    reward_model.eval()

    # Train with RLOO
    train_rloo(args, policy_model, reward_model, tokenizer, device, prompt_dataloader, validation_dataloader, run)

    # Evaluate final model
    samples = evaluate_rloo(args, policy_model, tokenizer, device, run)

    with open("rloo_samples.json", "w") as f:
        json.dump(samples, f, indent=2)

    save_checkpoint(policy_model, tokenizer, epoch=args.num_epochs)

    run.finish()

def train_bradley_terry_reward(args, tokenizer, device, run):
    """Train a Bradley-Terry reward model on preference data"""
    print("Loading preference data for reward model training...")
    
    # Load preference dataset
    pref_train_loader = get_preference_dataloader(args, "train", tokenizer)
    pref_val_loader = get_preference_dataloader(args, "val", tokenizer)
    
    # Initialize reward model
    reward_model = BradleyTerryRewardModel(args.sft_checkpoint)
    reward_model = reward_model.to(device)
    
    optimizer = optim.Adam(reward_model.parameters(), lr=args.reward_learning_rate)
    
    best_val_loss = float('inf')
    
    for epoch in range(args.reward_epochs):
        reward_model.train()
        total_loss = 0
        
        # Create progress bar outside the loop
        pbar = tqdm(total=len(pref_train_loader), desc=f"Reward Epoch {epoch}", leave=False)
        
        for batch_idx, batch in enumerate(pref_train_loader):
            batch = {k: v.to(device) for k, v in batch.items()}
            
            # Get rewards for chosen and rejected
            chosen_rewards = reward_model(
                batch["chosen_input_ids"],
                batch["chosen_attention_mask"]
            )
            rejected_rewards = reward_model(
                batch["rejected_input_ids"],
                batch["rejected_attention_mask"]
            )
            
            # Bradley-Terry loss
            loss = -F.logsigmoid(chosen_rewards - rejected_rewards).mean()
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            # Update progress bar
            pbar.update(1)
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        pbar.close()
        
        avg_train_loss = total_loss / len(pref_train_loader)
        
        # Validation
        reward_model.eval()
        val_loss = 0
        val_accuracy = 0
        
        with torch.no_grad():
            for batch in pref_val_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                
                chosen_rewards = reward_model(
                    batch["chosen_input_ids"],
                    batch["chosen_attention_mask"]
                )
                rejected_rewards = reward_model(
                    batch["rejected_input_ids"],
                    batch["rejected_attention_mask"]
                )
                
                loss = -F.logsigmoid(chosen_rewards - rejected_rewards).mean()
                accuracy = (chosen_rewards > rejected_rewards).float().mean()
                
                val_loss += loss.item()
                val_accuracy += accuracy.item()
        
        avg_val_loss = val_loss / len(pref_val_loader)
        avg_val_accuracy = val_accuracy / len(pref_val_loader)
        
        print(f"Reward Epoch {epoch}: Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Val Acc: {avg_val_accuracy:.4f}")
        
        run.log({
            "reward_train_loss": avg_train_loss,
            "reward_val_loss": avg_val_loss,
            "reward_val_accuracy": avg_val_accuracy
        })
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
    
    return reward_model

def sample_from_policy_batched(policy_model, prompts, tokenizer, device, num_samples=2, temperature=1.0, max_length=128):
    """Generate for all prompts in a single batch - much faster than sequential generation"""
    policy_model.eval()
    
    # Set padding side to left for decoder-only models
    original_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    
    # Format all prompts
    formatted_prompts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True
        )
        for prompt in prompts
    ]
    
    # Batch tokenize with padding
    inputs = tokenizer(
        formatted_prompts, 
        return_tensors="pt", 
        truncation=True, 
        max_length=1024,
        padding=True
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Track original lengths for extracting responses
    input_lengths = inputs["attention_mask"].sum(dim=1).tolist()
    
    all_samples = [[] for _ in prompts]
    
    # Generate num_samples times
    with torch.no_grad():
        for sample_idx in range(num_samples):
            outputs = policy_model.generate(
                **inputs,
                max_new_tokens=max_length,
                temperature=temperature,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id
            )
            
            # Extract responses for each prompt
            for prompt_idx, (output, input_len) in enumerate(zip(outputs, input_lengths)):
                # Get only the generated part
                generated_tokens = output[input_len:]
                generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
                full_text = tokenizer.decode(output, skip_special_tokens=True)
                
                all_samples[prompt_idx].append({
                    "prompt": prompts[prompt_idx],
                    "response": generated_text,
                    "full_text": full_text
                })
    
    # Restore original padding side
    tokenizer.padding_side = original_padding_side
    
    return all_samples

def compute_rloo_loss(policy_model, reward_model, samples, tokenizer, device):
    """Compute RLOO loss with leave-one-out baseline"""
    policy_model.train()
    
    all_losses = []
    all_rewards = []
    
    for prompt_samples in samples:
        # Get rewards for all samples
        rewards = []
        log_probs = []
        
        for sample in prompt_samples:
            # Tokenize the full text
            inputs = tokenizer(
                sample["full_text"],
                return_tensors="pt",
                truncation=True,
                max_length=512
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            # Get reward
            with torch.no_grad():
                reward = reward_model(inputs["input_ids"], inputs["attention_mask"])
            rewards.append(reward.item())
            
            # Get log probability from policy
            outputs = policy_model(**inputs, labels=inputs["input_ids"])
            log_prob = -outputs.loss.item()  # Negative loss is log probability
            log_probs.append(log_prob)
        
        rewards = torch.tensor(rewards, device=device)
        log_probs = torch.tensor(log_probs, device=device, requires_grad=True)
        
        # Compute RLOO baseline (leave-one-out mean)
        num_samples = len(prompt_samples)
        rloo_losses = []
        
        for i in range(num_samples):
            # Leave-one-out baseline
            baseline = (rewards.sum() - rewards[i]) / (num_samples - 1)
            advantage = rewards[i] - baseline
            
            # RLOO loss for this sample
            loss = -advantage * log_probs[i]
            rloo_losses.append(loss)
        
        # Average loss for this prompt
        prompt_loss = torch.stack(rloo_losses).mean()
        all_losses.append(prompt_loss)
        all_rewards.append(rewards.mean().item())
    
    # Return average loss across all prompts
    return torch.stack(all_losses).mean(), np.mean(all_rewards)

def train_rloo(args, policy_model, reward_model, tokenizer, device, prompt_dataloader, validation_dataloader, run):
    print("Training with RLOO")
    optimizer = optim.Adam(policy_model.parameters(), lr=args.learning_rate)
    
    num_training_steps = args.num_epochs * len(prompt_dataloader)
    num_warmup_steps = int(0.1 * num_training_steps)

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps
    )

    best_val_reward = -float('inf')
    
    for epoch in range(args.num_epochs):
        policy_model.train()
        epoch_loss = 0
        epoch_reward = 0
        
        pbar = tqdm(total=len(prompt_dataloader), desc=f"RLOO Epoch {epoch}", leave=False, position=0)
        
        for i, batch in enumerate(prompt_dataloader):
            prompts = batch["prompts"]
            
            # Sample from policy using batched HuggingFace generate
            samples = sample_from_policy_batched(
                policy_model, prompts, tokenizer, device,
                num_samples=args.num_samples,
                temperature=args.temperature,
                max_length=128  # Keep it short for speed
            )
            
            # Compute RLOO loss
            optimizer.zero_grad()
            
            # Use batched gradient computation
            batch_loss, batch_reward = compute_rloo_gradient_batch(
                policy_model, reward_model, samples, 
                tokenizer, device, optimizer
            )
            
            # Gradient step
            torch.nn.utils.clip_grad_norm_(policy_model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            
            epoch_loss += batch_loss
            epoch_reward += batch_reward
            
            # Update progress bar
            pbar.update(1)
            pbar.set_postfix({'loss': f'{batch_loss:.4f}', 'reward': f'{batch_reward:.4f}'})
            
            if i % 10 == 0:
                run.log({
                    "rloo_loss": batch_loss,
                    "rloo_reward": batch_reward,
                    "learning_rate": scheduler.get_last_lr()[0]
                })
            
            # Clear cache periodically
            if i % 50 == 0:
                torch.cuda.empty_cache()
        
        pbar.close()
        
        avg_loss = epoch_loss / len(prompt_dataloader)
        avg_reward = epoch_reward / len(prompt_dataloader)
        
        print(f"Epoch {epoch}: Avg Loss: {avg_loss:.4f}, Avg Reward: {avg_reward:.4f}")
        
        # Validation
        val_reward = validate_rloo(policy_model, reward_model, tokenizer, device, validation_dataloader, args)
        
        run.log({
            "epoch": epoch,
            "avg_train_loss": avg_loss,
            "avg_train_reward": avg_reward,
            "val_reward": val_reward
        })
        
        # Save best model
        if val_reward > best_val_reward:
            best_val_reward = val_reward
            save_checkpoint(policy_model, tokenizer, epoch=f"rloo_best_epoch_{epoch}")

def compute_rloo_gradient_batch(policy_model, reward_model, all_samples, tokenizer, device, optimizer):
    """Compute RLOO gradient for multiple prompts at once"""
    all_texts = []
    sample_to_prompt_idx = []
    samples_per_prompt = []
    
    # Flatten all samples
    for prompt_idx, prompt_samples in enumerate(all_samples):
        all_texts.extend([s["full_text"] for s in prompt_samples])
        sample_to_prompt_idx.extend([prompt_idx] * len(prompt_samples))
        samples_per_prompt.append(len(prompt_samples))
    
    # Batch tokenize all samples
    batch_size = 8  # Process in smaller chunks to avoid OOM
    all_rewards = []
    
    for i in range(0, len(all_texts), batch_size):
        batch_texts = all_texts[i:i+batch_size]
        inputs = tokenizer(
            batch_texts,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            rewards = reward_model(inputs["input_ids"], inputs["attention_mask"])
            all_rewards.extend(rewards.float().cpu().numpy())
    
    # Compute RLOO gradients
    total_loss = 0
    total_reward = 0
    prompt_idx = 0
    sample_idx = 0
    
    for prompt_samples, num_samples in zip(all_samples, samples_per_prompt):
        prompt_rewards = all_rewards[sample_idx:sample_idx + num_samples]
        
        # Compute advantages
        for i in range(num_samples):
            baseline = (sum(prompt_rewards) - prompt_rewards[i]) / (num_samples - 1) if num_samples > 1 else 0
            advantage = prompt_rewards[i] - baseline
            
            # Get log prob
            text = prompt_samples[i]["full_text"]
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            # Mask prompt tokens
            prompt_text = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt_samples[i]["prompt"]}],
                tokenize=False,
                add_generation_prompt=True
            )
            prompt_len = len(tokenizer(prompt_text)["input_ids"])
            
            labels = inputs["input_ids"].clone()
            labels[0, :prompt_len] = -100
            
            outputs = policy_model(**inputs, labels=labels)
            log_prob = -outputs.loss
            
            loss = -advantage * log_prob
            loss.backward()
            
            total_loss += loss.item()
        
        total_reward += np.mean(prompt_rewards)
        sample_idx += num_samples
    
    return total_loss / len(all_samples), total_reward / len(all_samples)

def validate_rloo(policy_model, reward_model, tokenizer, device, dataloader, args):
    """Validate RLOO by sampling and computing average reward"""
    policy_model.eval()
    total_reward = 0
    num_prompts = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validation"):
            prompts = batch["prompts"]
            
            # Sample from policy using batched generation
            samples = sample_from_policy_batched(
                policy_model, prompts, tokenizer, device,
                num_samples=1,  # Single sample for validation
                temperature=args.temperature,
                max_length=128
            )
            
            # Compute rewards
            for prompt_samples in samples:
                for sample in prompt_samples:
                    inputs = tokenizer(
                        sample["full_text"],
                        return_tensors="pt",
                        truncation=True,
                        max_length=512
                    )
                    inputs = {k: v.to(device) for k, v in inputs.items()}
                    
                    reward = reward_model(inputs["input_ids"], inputs["attention_mask"])
                    total_reward += reward.item()
                    num_prompts += 1
    
    return total_reward / num_prompts if num_prompts > 0 else 0

def evaluate_rloo(args, model, tokenizer, device, run, num_samples=50):
    """Evaluate RLOO model by generating samples"""
    ds = load_dataset(args.dataset_name, split="test_prefs")
    ds = ds.select(range(0, num_samples))
    
    prompts = [example["prompt"] for example in ds]
    
    # Sample from model
    model.eval()
    generations = []
    
    for prompt in tqdm(prompts, desc="Generating samples"):
        formatted_prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True
        )
        
        inputs = tokenizer(formatted_prompt, return_tensors="pt", truncation=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        input_length = inputs["input_ids"].shape[1]
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=350,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id
            )
        
        generated_tokens = outputs[0][input_length:]
        generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
        
        generations.append({
            "prompt": prompt,
            "response": generated_text
        })
    
    # Log to wandb
    run.log({
        "rloo_samples": wandb.Table(
            columns=["prompt", "generation"],
            data=[[g["prompt"], g["response"]] for g in generations[:10]]
        )
    })
    
    return generations

def save_checkpoint(model, tokenizer, epoch, checkpoint_dir="./checkpoints"):
    os.makedirs(checkpoint_dir, exist_ok=True)

    checkpoint_path = os.path.join(checkpoint_dir, f"{epoch}")
    model.save_pretrained(checkpoint_path)
    tokenizer.save_pretrained(checkpoint_path)
    print(f"Checkpoint saved to: {checkpoint_path}")

def get_prompt_dataloader(args, split):
    """Get dataloader for prompts only (for on-policy sampling)"""
    dataset_name = args.dataset_name

    if split == "train":
        ds = load_dataset(dataset_name, split="train_prefs")
        ds = ds.select(range(2000, min(len(ds), 12000)))
    elif split == "val":
        ds = load_dataset(dataset_name, split="train_prefs") 
        ds = ds.select(range(0, 2000))
    else:
        ds = load_dataset(dataset_name, split="test_prefs")

    # Extract just the prompts
    prompts = [example["prompt"] for example in ds]
    
    # Create batches of prompts
    def collate_fn(batch):
        return {"prompts": batch}
    
    from torch.utils.data import DataLoader
    dataloader = DataLoader(
        prompts,
        batch_size=args.batch_size,
        collate_fn=collate_fn,
        shuffle=(split == "train")
    )
    
    return dataloader

def get_preference_dataloader(args, split, tokenizer):
    """Get preference dataloader for Bradley-Terry training"""
    dataset_name = args.dataset_name

    if split == "train":
        ds = load_dataset(dataset_name, split="train_prefs")
        ds = ds.select(range(14000, 20000))
    elif split == "val":
        ds = load_dataset(dataset_name, split="train_prefs")
        ds = ds.select(range(12000, 14000))
    else:
        ds = load_dataset(dataset_name, split="test_prefs")

    def preprocess(example):
        # Format full conversations
        chosen_messages = [
            {"role": "user", "content": example["prompt"]},
            {"role": "assistant", "content": example["chosen"][1]["content"]}
        ]
        chosen_text = tokenizer.apply_chat_template(chosen_messages, tokenize=False)
        
        rejected_messages = [
            {"role": "user", "content": example["prompt"]},
            {"role": "assistant", "content": example["rejected"][1]["content"]}
        ]
        rejected_text = tokenizer.apply_chat_template(rejected_messages, tokenize=False)
        
        # Tokenize
        chosen_encoded = tokenizer(chosen_text, truncation=True, max_length=512)
        rejected_encoded = tokenizer(rejected_text, truncation=True, max_length=512)
        
        return {
            "chosen_input_ids": chosen_encoded["input_ids"],
            "chosen_attention_mask": chosen_encoded["attention_mask"],
            "rejected_input_ids": rejected_encoded["input_ids"],
            "rejected_attention_mask": rejected_encoded["attention_mask"],
        }
    
    preprocessed_ds = ds.map(preprocess, num_proc=4)
    
    def collate_fn(batch):
        # Find max length
        max_len = max(
            max(len(x["chosen_input_ids"]) for x in batch),
            max(len(x["rejected_input_ids"]) for x in batch)
        )
        
        # Pad all sequences to same length
        def pad_to_length(sequences, padding_value, max_len):
            padded = []
            for seq in sequences:
                seq_tensor = torch.tensor(seq) if not isinstance(seq, torch.Tensor) else seq
                if len(seq_tensor) < max_len:
                    padding = torch.full((max_len - len(seq_tensor),), padding_value)
                    padded_seq = torch.cat([seq_tensor, padding])
                else:
                    padded_seq = seq_tensor[:max_len]
                padded.append(padded_seq)
            return torch.stack(padded)
        
        chosen_input_ids = pad_to_length([x["chosen_input_ids"] for x in batch], tokenizer.pad_token_id, max_len)
        chosen_attention_mask = pad_to_length([x["chosen_attention_mask"] for x in batch], 0, max_len)
        rejected_input_ids = pad_to_length([x["rejected_input_ids"] for x in batch], tokenizer.pad_token_id, max_len)
        rejected_attention_mask = pad_to_length([x["rejected_attention_mask"] for x in batch], 0, max_len)
        
        return {
            "chosen_input_ids": chosen_input_ids,
            "chosen_attention_mask": chosen_attention_mask,
            "rejected_input_ids": rejected_input_ids,
            "rejected_attention_mask": rejected_attention_mask,
        }
    
    dataloader = torch.utils.data.DataLoader(
        preprocessed_ds,
        batch_size=args.batch_size,
        collate_fn=collate_fn,
        shuffle=(split == "train")
    )
    
    return dataloader

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sft_checkpoint", type=str, default="/home/ec2-user/cs224r_final/checkpoints/dpo_best_epoch_2")
    parser.add_argument("--dataset_name", type=str, default="HuggingFaceH4/ultrafeedback_binarized")
    parser.add_argument("--batch_size", type=int, default=32, help="Increased for batched generation efficiency")
    parser.add_argument("--learning_rate", type=float, default=1e-6)
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--num_samples", type=int, default=4, help="Reduced from 4 for speed")
    parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature")
    parser.add_argument("--seed", type=int, default=42)
    
    # Reward model args
    parser.add_argument("--reward_model_path", type=str, default=None, help="Path to pre-trained reward model")
    parser.add_argument("--reward_learning_rate", type=float, default=5e-5)
    parser.add_argument("--reward_epochs", type=int, default=3)
    
    return parser.parse_args()

if __name__ == "__main__":
    main()