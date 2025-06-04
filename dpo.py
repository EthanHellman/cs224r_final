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

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

        # model_name = "Qwen/Qwen2.5-0.5B"
# dataset_name = "HuggingFaceH4/ultrafeedback_binarized"
# num_epochs = 10

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
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "beta": args.beta,
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

    train_dataloader = get_dataloader(args, "train")
    validation_dataloader = get_dataloader(args, "val")

    model = AutoModelForCausalLM.from_pretrained(
        args.sft_checkpoint,
        torch_dtype="auto",
        device_map="auto"
    )

    ref_model = AutoModelForCausalLM.from_pretrained(
        args.sft_checkpoint,
        torch_dtype="auto",
        device_map="auto"
    )

    # Freeze reference model
    for param in ref_model.parameters():
        param.requires_grad = False

    model.gradient_checkpointing_enable()

    train(args, model, ref_model, tokenizer, device, train_dataloader, validation_dataloader, run)

    samples = evaluate_dpo(args, model, tokenizer, device, run)

    with open("dpo_samples.json", "w") as f:
        json.dump(samples, f, indent=2)

    save_checkpoint(model, tokenizer, epoch=args.num_epochs)

    run.finish()

def compute_dpo_loss(model, ref_model, batch, device, beta):
    # Get model outputs for chosen and rejected (without labels to avoid loss computation)
    chosen_outputs = model(
        input_ids=batch["chosen_input_ids"],
        attention_mask=batch["chosen_attention_mask"]
    )
    
    rejected_outputs = model(
        input_ids=batch["rejected_input_ids"],
        attention_mask=batch["rejected_attention_mask"]
    )
    
    # Get reference model outputs
    with torch.no_grad():
        ref_chosen_outputs = ref_model(
            input_ids=batch["chosen_input_ids"],
            attention_mask=batch["chosen_attention_mask"]
        )
        
        ref_rejected_outputs = ref_model(
            input_ids=batch["rejected_input_ids"],
            attention_mask=batch["rejected_attention_mask"]
        )
    
    # Calculate log probs manually
    def get_logprobs(logits, labels):
        # Shift so that tokens < n predict n
        logits = logits[:, :-1, :].contiguous()
        labels = labels[:, 1:].contiguous()
        
        # Get log probs
        log_probs = F.log_softmax(logits, dim=-1)
        
        # Create mask for valid tokens (not -100)
        mask = (labels != -100)
        
        # Replace -100 with 0 for gather (will be masked out anyway)
        labels_for_gather = labels.clone()
        labels_for_gather[~mask] = 0
        
        # Gather the log probs for the labels
        labels_for_gather = labels_for_gather.unsqueeze(-1)
        log_probs = torch.gather(log_probs, -1, labels_for_gather).squeeze(-1)
        
        # Apply mask to zero out padding positions
        log_probs = log_probs * mask.float()
        
        # Sum over sequence for each example
        return log_probs.sum(dim=1)
    
    chosen_logprobs = get_logprobs(chosen_outputs.logits, batch["chosen_labels"])
    rejected_logprobs = get_logprobs(rejected_outputs.logits, batch["rejected_labels"])
    ref_chosen_logprobs = get_logprobs(ref_chosen_outputs.logits, batch["chosen_labels"])
    ref_rejected_logprobs = get_logprobs(ref_rejected_outputs.logits, batch["rejected_labels"])
    
    # DPO loss calculation
    chosen_rewards = beta * (chosen_logprobs - ref_chosen_logprobs)
    rejected_rewards = beta * (rejected_logprobs - ref_rejected_logprobs)
    
    # Average over batch
    loss = -F.logsigmoid(chosen_rewards - rejected_rewards).mean()
    
    # Calculate accuracy
    accuracy = (chosen_rewards > rejected_rewards).float().mean()
    
    return loss, accuracy, chosen_rewards.mean(), rejected_rewards.mean()

def train(args, model, ref_model, tokenizer, device, train_dataloader, validation_dataloader, run):
    print("Training DPO")
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    gradient_accumulation_steps = args.gradient_accumulation_steps
    
    num_training_steps = args.num_epochs * len(train_dataloader) // gradient_accumulation_steps
    num_warmup_steps = int(0.1 * num_training_steps)

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps
    )

    model.train()
    train_losses = []

    best_val_accuracy = 0
    patience = 3
    patience_counter = 0
    
    for epoch in range(args.num_epochs):
        model.train()
        epoch_loss = 0
        epoch_accuracy = 0
        optimizer.zero_grad()

        for i, batch in enumerate(train_dataloader):
            batch = {k: v.to(device) for k, v in batch.items()}

            with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
                loss, accuracy, chosen_rewards, rejected_rewards = compute_dpo_loss(
                    model, ref_model, batch, device, args.beta
                )
                loss = loss / gradient_accumulation_steps
            
            loss.backward()

            if i % 10 == 0:
                curr_lr = scheduler.get_last_lr()[0]
                actual_loss = loss.item() * gradient_accumulation_steps

                print(f"Epoch: {epoch}, Batch: {i}, Loss: {actual_loss:.4f}, Accuracy: {accuracy.item():.4f}, Current LR: {curr_lr:.2e}")

                run.log({
                    "loss": actual_loss, 
                    "accuracy": accuracy.item(),
                    "learning_rate": curr_lr,
                    "chosen_rewards": chosen_rewards.item(),
                    "rejected_rewards": rejected_rewards.item()
                })

            if (i + 1) % gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                total_norm = 0

                for p in model.parameters():
                    if p.grad is not None:
                        param_norm = p.grad.data.norm(2)
                        total_norm += param_norm.item() ** 2
                total_norm = total_norm ** 0.5

                run.log({
                    "gradient_norm": total_norm
                })

                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            epoch_loss += loss.item() * gradient_accumulation_steps
            epoch_accuracy += accuracy.item()
            
            del loss, accuracy
            if i % 50 == 0:
                torch.cuda.empty_cache()

        if len(train_dataloader) % gradient_accumulation_steps != 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            
        avg_loss = epoch_loss / len(train_dataloader)
        avg_accuracy = epoch_accuracy / len(train_dataloader)

        train_losses.append(avg_loss)
        print(f"Epoch: {epoch}, Average Loss: {avg_loss}, Average Accuracy: {avg_accuracy}")

        val_loss, val_accuracy = validate_dpo(model, ref_model, device, validation_dataloader, args.beta)

        run.log({
            "avg_train_loss": avg_loss,
            "avg_train_accuracy": avg_accuracy,
            "val_loss": val_loss,
            "val_accuracy": val_accuracy
        })

        print(f"Epoch: {epoch}, Validation Loss: {val_loss}, Validation Accuracy: {val_accuracy}")

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            patience_counter = 0
            save_checkpoint(model, tokenizer, epoch=f"dpo_best_epoch_{epoch}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch}")
                break
    
    return train_losses

def validate_dpo(model, ref_model, device, dataloader, beta):
    model.eval()
    total_loss = 0
    total_accuracy = 0

    with torch.no_grad():
        for batch in dataloader:
            batch = {k: v.to(device) for k, v in batch.items()} 

            with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
                loss, accuracy, _, _ = compute_dpo_loss(
                    model, ref_model, batch, device, beta
                )

            total_loss += loss.item()
            total_accuracy += accuracy.item()

    val_loss = total_loss / len(dataloader)
    val_accuracy = total_accuracy / len(dataloader)

    return val_loss, val_accuracy

def generate_samples(model, tokenizer, prompts, device, max_length=350):
    model.eval()
    samples = []

    for prompt in prompts:
        formatted_prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt["prompt"]}],
            tokenize=False,
            add_generation_prompt=True
        )

        inputs = tokenizer(formatted_prompt, return_tensors="pt", truncation=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        input_length = inputs["input_ids"].shape[1]

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_length,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id
            )
        
        generated_tokens = outputs[0][input_length:]
        generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)

        samples.append({
            "prompt": prompt["prompt"],
            "response": generated_text,
            "chosen": prompt["chosen"],
            "rejected": prompt["rejected"]
        })
    
    return samples

def evaluate_dpo(args, model, tokenizer, device, run, num_samples=50):
    ds = load_dataset(args.dataset_name, split="test_prefs")
    ds = ds.select(range(0, num_samples))

    prompts = [{
        "prompt": example["prompt"], 
        "chosen": example["chosen"][1]["content"],
        "rejected": example["rejected"][1]["content"]
    } for example in ds]

    generations = generate_samples(model, tokenizer, prompts, device)

    run.log({
        "samples": wandb.Table(
            columns=["prompt", "generation", "chosen", "rejected"],
            data=[[e["prompt"], e["response"], e["chosen"], e["rejected"]] for e in generations]
        )
    })

    return generations

def save_checkpoint(model, tokenizer, epoch, checkpoint_dir="./checkpoints"):
    os.makedirs(checkpoint_dir, exist_ok=True)

    checkpoint_path = os.path.join(checkpoint_dir, f"{epoch}")
    model.save_pretrained(checkpoint_path)
    tokenizer.save_pretrained(checkpoint_path)
    print(f"Checkpoint saved to: {checkpoint_path}")

def get_dataloader(args, split):
    dataset_name = args.dataset_name

    if split == "train":
        ds = load_dataset(dataset_name, split="train_prefs")
        ds = ds.select(range(2000, min(len(ds), 12000)))
    elif split == "val":
        ds = load_dataset(dataset_name, split="train_prefs")
        ds = ds.select(range(0, 2000))
    else:
        ds = load_dataset(dataset_name, split="test_prefs")

    tokenizer = AutoTokenizer.from_pretrained(args.sft_checkpoint)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    def preprocess(example):
        # Process chosen
        chosen_messages = [
            {"role": "user", "content": example["prompt"]},
            {"role": "assistant", "content": example["chosen"][1]["content"]}
        ]
        chosen = tokenizer.apply_chat_template(
            chosen_messages,
            tokenize=False,
            add_generation_prompt=False
        )

        # Process rejected
        rejected_messages = [
            {"role": "user", "content": example["prompt"]},
            {"role": "assistant", "content": example["rejected"][1]["content"]}
        ]
        rejected = tokenizer.apply_chat_template(
            rejected_messages,
            tokenize=False,
            add_generation_prompt=False
        )

        # Get prompt for masking
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": example["prompt"]}],
            tokenize=False,
            add_generation_prompt=True
        )

        prompt_ids = tokenizer(prompt, return_tensors=None, truncation=True, padding=False)["input_ids"]
        
        # Tokenize full sequences
        chosen_full = tokenizer(chosen, return_tensors=None, truncation=True, padding=False, max_length=512)
        rejected_full = tokenizer(rejected, return_tensors=None, truncation=True, padding=False, max_length=512)

        # Create labels
        chosen_labels = chosen_full["input_ids"].copy()
        chosen_labels[:len(prompt_ids)] = [-100] * len(prompt_ids)

        rejected_labels = rejected_full["input_ids"].copy()
        rejected_labels[:len(prompt_ids)] = [-100] * len(prompt_ids)

        return {
            "chosen_input_ids": chosen_full["input_ids"],
            "chosen_attention_mask": chosen_full["attention_mask"],
            "chosen_labels": chosen_labels,
            "rejected_input_ids": rejected_full["input_ids"],
            "rejected_attention_mask": rejected_full["attention_mask"],
            "rejected_labels": rejected_labels
        }

    preprocessed_ds = ds.map(preprocess)
    preprocessed_ds.set_format(type="torch", columns=[
        "chosen_input_ids", "chosen_attention_mask", "chosen_labels",
        "rejected_input_ids", "rejected_attention_mask", "rejected_labels"
    ])

    def collate_fn(batch):
        chosen_input_ids = []
        chosen_attention_mask = []
        chosen_labels = []
        rejected_input_ids = []
        rejected_attention_mask = []
        rejected_labels = []

        for item in batch:
            chosen_input_ids.append(item["chosen_input_ids"])
            chosen_attention_mask.append(item["chosen_attention_mask"])
            chosen_labels.append(item["chosen_labels"])
            rejected_input_ids.append(item["rejected_input_ids"])
            rejected_attention_mask.append(item["rejected_attention_mask"])
            rejected_labels.append(item["rejected_labels"])
        
        # Find max length across both chosen and rejected
        max_len = max(
            max(len(x) for x in chosen_input_ids),
            max(len(x) for x in rejected_input_ids)
        )
        
        # Pad all sequences to the same max length
        def pad_to_length(sequences, padding_value, max_len):
            padded = []
            for seq in sequences:
                seq_tensor = seq if isinstance(seq, torch.Tensor) else torch.tensor(seq)
                if len(seq_tensor) < max_len:
                    padding = torch.full((max_len - len(seq_tensor),), padding_value, dtype=seq_tensor.dtype)
                    padded_seq = torch.cat([seq_tensor, padding])
                else:
                    padded_seq = seq_tensor[:max_len]
                padded.append(padded_seq)
            return torch.stack(padded)
        
        chosen_input_ids = pad_to_length(chosen_input_ids, tokenizer.pad_token_id, max_len)
        chosen_attention_mask = pad_to_length(chosen_attention_mask, 0, max_len)
        chosen_labels = pad_to_length(chosen_labels, -100, max_len)
        
        rejected_input_ids = pad_to_length(rejected_input_ids, tokenizer.pad_token_id, max_len)
        rejected_attention_mask = pad_to_length(rejected_attention_mask, 0, max_len)
        rejected_labels = pad_to_length(rejected_labels, -100, max_len)
        
        return {
            "chosen_input_ids": chosen_input_ids,
            "chosen_attention_mask": chosen_attention_mask,
            "chosen_labels": chosen_labels,
            "rejected_input_ids": rejected_input_ids,
            "rejected_attention_mask": rejected_attention_mask,
            "rejected_labels": rejected_labels
        }

    dataloader = torch.utils.data.DataLoader(
        preprocessed_ds,
        batch_size=args.batch_size,
        collate_fn=collate_fn
    )

    print("_" * 50)

    return dataloader

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sft_checkpoint", type=str, default="/home/ec2-user/cs224r_final/checkpoints/sft_epoch_best_epoch_1")
    parser.add_argument("--dataset_name", type=str, default="HuggingFaceH4/ultrafeedback_binarized")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=2)
    parser.add_argument("--learning_rate", type=float, default=5e-6)
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()

if __name__ == "__main__":
    main()