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
import wandb
import argparse
import json
import os
import random  # Add random module for debugging sampling
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
def main():
    args = parse_args()
    run = wandb.init(
        entity="ethanhellman",
        project="CS224R",
        config={
            "learning_rate": args.learning_rate,
            "model": args.model_name,
            "dataset": args.dataset_name,
            "epochs": args.num_epochs,
            "batch_size": args.batch_size,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "seed": args.seed,
            "validation_interval": args.validation_interval,
            "sample_generation_interval": args.sample_generation_interval
        }
    )
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"CUDA device count: {torch.cuda.device_count()}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    train_dataloader = get_dataloader(args, "train")
    validation_dataloader = get_dataloader(args, "val")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype="auto",
        device_map="auto"
    )
    model.gradient_checkpointing_enable()
    train(args, model, tokenizer, device, train_dataloader, validation_dataloader, run)
    samples = evaluate_feedback(args, model, tokenizer, device, run)
    bleu_scores = calculate_bleu_score(samples)
    print(f"BLEU-1: {bleu_scores['bleu_1']:.4f}")
    print(f"BLEU-2: {bleu_scores['bleu_2']:.4f}")
    print(f"BLEU-4: {bleu_scores['bleu_4']:.4f}")
    run.log(bleu_scores)
    with open("sft_samples.json", "w") as f:
        json.dump(samples, f, indent=2)
    save_checkpoint(model, tokenizer, name="final")
    run.finish()
def train(args, model, tokenizer, device, train_dataloader, validation_dataloader, run):
    print("Training")
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
    best_val_loss = float('inf')
    patience = 2
    patience_counter = 0
    steps_without_improvement = 0
    
    global_step = 0
    
    # Initialize the length penalty to 0
    current_length_penalty = torch.tensor(0.0, device=device)
    
    # Select a few fixed examples for length checking
    length_check_examples = []
    ds = load_dataset(args.dataset_name, split="train_sft")
    for i in range(3):  # Use 3 examples
        idx = random.randint(0, len(ds) - 1)
        example = ds[idx]
        prompt = example["prompt"]
        target = example["chosen"][1]["content"]
        target_length = len(tokenizer(target, return_tensors=None)["input_ids"])
        length_check_examples.append({
            "prompt": prompt,
            "target": target,
            "target_length": target_length
        })
    
    for epoch in range(args.num_epochs):
        model.train()
        epoch_loss = 0
        optimizer.zero_grad()
        for i, batch in enumerate(train_dataloader):
            batch = {k: v.to(device) for k, v in batch.items()}
            # Update length penalty periodically
            if args.length_penalty_interval > 0 and global_step % args.length_penalty_interval == 0:
                # Generate samples and compute length penalty
                current_length_penalty = compute_generation_length_penalty(
                    model, tokenizer, length_check_examples, device, args.length_penalty_factor
                )
                
                # Log the length penalty
                run.log({
                    "generation_length_penalty": current_length_penalty.item()
                })
            with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
                outputs = model(**batch)
                lm_loss = outputs.loss
                
                # Apply the current length penalty to every step
                # Use the most recently calculated penalty value
                loss = lm_loss + current_length_penalty
                
                # Scale for gradient accumulation
                loss = loss / gradient_accumulation_steps
            
            loss.backward()
            if i % 10 == 0:
                curr_lr = scheduler.get_last_lr()[0]
                actual_loss = loss.item() * gradient_accumulation_steps
                lm_loss_value = lm_loss.item()
                penalty_value = current_length_penalty.item()
                print(f"Epoch: {epoch}, Batch: {i}, Loss: {actual_loss:.4f}, LM Loss: {lm_loss_value:.4f}, Length Penalty: {penalty_value:.4f}, Current LR: {curr_lr:.2e}")
                run.log({
                    "loss": actual_loss, 
                    "lm_loss": lm_loss_value,
                    "length_penalty": penalty_value,
                    "learning_rate": curr_lr,
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
                global_step += 1
                
                # Run validation at specified intervals
                if args.validation_interval > 0 and global_step % args.validation_interval == 0:
                    print(f"\nRunning validation at step {global_step}")
                    val_loss, perplexity = validate(model, device, validation_dataloader)
                    print(f"Validation Loss: {val_loss:.4f}, Perplexity: {perplexity:.4f}")
                    
                    run.log({
                        "interim_val_loss": val_loss,
                        "interim_val_perplexity": perplexity
                    })
                    
                    # Check if this is the best model so far (based on validation loss)
                    if val_loss < best_val_loss:
                        improvement = best_val_loss - val_loss
                        best_val_loss = val_loss
                        steps_without_improvement = 0
                        
                        # Save the best model checkpoint
                        save_checkpoint(model, tokenizer, name=f"best_step_{global_step}")
                        print(f"New best model saved! Validation loss improved by {improvement:.4f}")
                    else:
                        steps_without_improvement += args.validation_interval
                        print(f"No improvement for {steps_without_improvement} steps (best: {best_val_loss:.4f})")
                        
                        # Check for early stopping based on validation steps
                        if steps_without_improvement >= args.early_stopping_steps:
                            print(f"Early stopping triggered after {steps_without_improvement} steps without improvement")
                            # Save final model before early stopping
                            save_checkpoint(model, tokenizer, name=f"final_early_stop_step_{global_step}")
                            return train_losses
                    
                    # Return to training mode
                    model.train()
                
                # Generate samples at specified intervals
                if args.sample_generation_interval > 0 and global_step % args.sample_generation_interval == 0:
                    print(f"\nGenerating samples at step {global_step}")
                    samples = generate_training_samples(args, model, tokenizer, device, run, num_samples=2)
                    
                    # Log samples to wandb
                    run.log({
                        f"training_samples_step_{global_step}": wandb.Table(
                            columns=["prompt", "generation", "reference"],
                            data=[[s["prompt"], s["response"], s["chosen"]] for s in samples]
                        )
                    })
                    
                    # Return to training mode
                    model.train()
            epoch_loss += loss.item() * gradient_accumulation_steps
            
            del outputs, loss
            if i % 50 == 0:
                torch.cuda.empty_cache()
        if len(train_dataloader) % gradient_accumulation_steps != 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            
        avg_loss = epoch_loss / len(train_dataloader)
        train_losses.append(avg_loss)
        print(f"Epoch: {epoch}, Average Loss: {avg_loss}")
        
        # End of epoch validation
        val_loss, perplexity = validate(model, device, validation_dataloader)
        run.log({
            "avg_train_loss": avg_loss,
            "val_loss": val_loss,
            "val_perplexity": perplexity
        })
        print(f"Epoch: {epoch}, Validation Loss: {val_loss}, Validation Perplexity: {perplexity}")
        
        # Save checkpoint for this epoch regardless of performance
        save_checkpoint(model, tokenizer, name=f"epoch_{epoch}")
        
        # Check if this is the best model so far (based on validation loss)
        if val_loss < best_val_loss:
            improvement = best_val_loss - val_loss
            best_val_loss = val_loss
            patience_counter = 0
            steps_without_improvement = 0
            
            # Save the best model checkpoint
            save_checkpoint(model, tokenizer, name=f"best_epoch_{epoch}")
            print(f"New best model saved! Validation loss improved by {improvement:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered after {patience} epochs without improvement")
                break
    
    if patience_counter > 0:
        print(f"Best model saved at checkpoint: best_epoch_{epoch - patience_counter}")
    else:
        print(f"Best model saved at checkpoint: best_epoch_{epoch}")
    return train_losses
def compute_generation_length_penalty(model, tokenizer, examples, device, penalty_factor=0.001):
    """
    Compute length penalty based on actual generated outputs
    """
    model.eval()  # Set to eval mode for generation
    length_diffs = []
    
    print("\nComputing generation length penalty...")
    
    with torch.no_grad():
        for example in examples:
            # Format the prompt
            formatted_prompt = tokenizer.apply_chat_template(
                [{"role": "user", "content": example["prompt"]}],
                tokenize=False,
                add_generation_prompt=True
            )
            
            # Encode the prompt
            inputs = tokenizer(formatted_prompt, return_tensors="pt").to(device)
            
            # Generate a response
            outputs = model.generate(
                **inputs,
                max_new_tokens=512,  # Allow enough tokens
                do_sample=True,
                temperature=0.9,
                pad_token_id=tokenizer.pad_token_id
            )
            
            # Decode the generated response (excluding the prompt)
            input_length = inputs.input_ids.shape[1]
            generated_text = tokenizer.decode(outputs[0][input_length:], skip_special_tokens=True)
            
            # Count tokens in the generated text
            generated_tokens = tokenizer(generated_text, return_tensors=None)["input_ids"]
            generated_length = len(generated_tokens)
            
            # Calculate difference from target (only penalize if too long)
            target_length = example["target_length"]
            length_diff = max(0, generated_length - target_length)
            length_diffs.append(length_diff)
            
            # Print details
            print(f"  Prompt: {example['prompt'][:50]}...")
            print(f"  Generated length: {generated_length}, Target length: {target_length}")
            print(f"  Difference: {length_diff}")
            print(f"  Sample of generation: {generated_text[:100]}...\n")
    
    # Calculate penalty (average of differences, not squared)
    if length_diffs:
        avg_diff = sum(length_diffs) / len(length_diffs)
        penalty = penalty_factor * avg_diff
    else:
        penalty = 0.0
    
    penalty_tensor = torch.tensor(penalty, device=device)
    print(f"Length penalty: {penalty:.4f}\n")
    
    model.train()  # Return to train mode
    return penalty_tensor
def generate_training_samples(args, model, tokenizer, device, run, num_samples=2):
    """Generate sample outputs during training to monitor quality using consistent test examples"""
    # Use the same examples from test split for consistent monitoring
    ds = load_dataset(args.dataset_name, split="test_sft")
    ds = ds.select(range(0, num_samples))
    
    prompts = [{"prompt": example["prompt"], "chosen": example["chosen"][1]["content"]} for example in ds]
    
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
                max_new_tokens=150,  # Shorter for training samples
                temperature=0.9,     # Increased temperature
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id
            )
        
        generated_tokens = outputs[0][input_length:]
        generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
        
        samples.append({
            "prompt": prompt["prompt"],
            "response": generated_text,
            "chosen": prompt["chosen"]  # Include reference response
        })
    
    return samples
def calculate_perplexity(model, dataloader, device):
    model.eval()
    total_loss = 0
    total_tokens = 0
    
    with torch.no_grad():
        for batch in dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            
            with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
                outputs = model(**batch)
                loss = outputs.loss
            
            # Count only non-padded tokens
            labels = batch["labels"]
            num_tokens = (labels != -100).sum().item()
            
            total_loss += loss.item() * num_tokens
            total_tokens += num_tokens
    
    avg_loss = total_loss / total_tokens
    perplexity = math.exp(avg_loss)
    
    return perplexity
def validate(model, device, dataloader):
    model.eval()
    total_loss = 0
    perplexity = calculate_perplexity(model, dataloader, device)
    with torch.no_grad():
        for batch in dataloader:
            batch = {k: v.to(device) for k, v in batch.items()} 
            with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
                outputs = model(**batch)
                loss = outputs.loss
            total_loss += loss.item()
    val_loss = total_loss / len(dataloader)
    return val_loss, perplexity
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
                temperature=0.9,  # Increased from 0.7 to reduce repetition
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id
            )
        
        generated_tokens = outputs[0][input_length:]
        generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
        samples.append({
            "prompt": prompt["prompt"],
            "response": generated_text,
            "chosen": prompt["chosen"]
        })
    
    return samples
def evaluate_feedback(args, model, tokenizer, device, run, num_samples=50):
    ds = load_dataset(args.dataset_name, split="test_sft")
    ds = ds.select(range(0, num_samples))
    prompts = [{"prompt": example["prompt"], "chosen": example["chosen"][1]["content"]} for example in ds]
    generations = generate_samples(model, tokenizer, prompts, device)
    run.log({
        "samples": wandb.Table(
            columns=["prompt", "generation", "chosen"],
            data=[[e["prompt"], e["response"], e["chosen"]] for e in generations]
        )
    })
    return generations
def save_checkpoint(model, tokenizer, name, checkpoint_dir="./checkpoints"):
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, f"sft_{name}")
    model.save_pretrained(checkpoint_path)
    tokenizer.save_pretrained(checkpoint_path)
    print(f"Checkpoint saved to: {checkpoint_path}")
def calculate_bleu_score(generations):
    bleu_1_scores = []
    bleu_2_scores = []
    bleu_4_scores = []
    smoother = SmoothingFunction()
    
    for gen in generations:
        reference = gen["chosen"].split()
        hypothesis = gen["response"].split()
        
        # BLEU-1 (unigram)
        bleu_1 = sentence_bleu([reference], hypothesis, 
                               weights=(1.0, 0, 0, 0),
                               smoothing_function=smoother.method1)
        bleu_1_scores.append(bleu_1)
        
        # BLEU-2 (unigram + bigram)
        bleu_2 = sentence_bleu([reference], hypothesis, 
                               weights=(0.5, 0.5, 0, 0),
                               smoothing_function=smoother.method1)
        bleu_2_scores.append(bleu_2)
        
        # BLEU-4 (standard)
        bleu_4 = sentence_bleu([reference], hypothesis,
                               smoothing_function=smoother.method1)
        bleu_4_scores.append(bleu_4)
    
    return {
        'bleu_1': sum(bleu_1_scores) / len(bleu_1_scores),
        'bleu_2': sum(bleu_2_scores) / len(bleu_2_scores),
        'bleu_4': sum(bleu_4_scores) / len(bleu_4_scores)
    }
def get_dataloader(args, split):
    dataset_name = args.dataset_name
    if split == "train":
        ds = load_dataset(dataset_name, split="train_sft")
        ds = ds.select(range(2000, len(ds)))
    elif split == "val":
        ds = load_dataset(dataset_name, split="train_sft")
        ds = ds.select(range(0, 2000))
    else:
        ds = load_dataset(dataset_name, split="test_sft")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    def preprocess(example):
        # Get the prompt and chosen response
        user_message = example["chosen"][0]["content"]
        chosen_response = example["chosen"][1]["content"]
        
        # Format as a chat template
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": user_message}],
            tokenize=False,
            add_generation_prompt=True
        )
        chosen = tokenizer.apply_chat_template(
            example["chosen"],
            tokenize=False,
            add_generation_prompt=False
        )
        # Get the token IDs
        prompt_ids = tokenizer(prompt, return_tensors=None, truncation=True, padding=False)["input_ids"]
        full_ids = tokenizer(chosen, return_tensors=None, truncation=True, padding=False)["input_ids"]
        
        # Calculate target length (from chosen response only, not including prompt)
        chosen_tokens = tokenizer(chosen_response, return_tensors=None)["input_ids"]
        target_length = len(chosen_tokens)
        
        # Create inputs and labels
        input_ids = full_ids
        attention_mask = [1] * len(full_ids)
        labels = full_ids.copy()
        
        # Mask out the prompt portion in labels
        labels[:len(prompt_ids)] = [-100] * len(prompt_ids)
        
        # For debugging - show a few examples randomly
        if random.random() < 0.001:  # Show ~0.1% of examples
            print(f"\nDEBUG EXAMPLE:")
            print(f"User: {user_message[:50]}...")
            print(f"Response: {chosen_response[:50]}...")
            print(f"Target length: {target_length}")
            print(f"Prompt length: {len(prompt_ids)}")
            print(f"Response portion: {len(labels) - len(prompt_ids)}")
            print(f"Labels [-100 count]: {labels.count(-100)}")
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "target_length": target_length,
            "prompt_length": len(prompt_ids)  # Save prompt length explicitly
        }
    preprocessed_ds = ds.map(preprocess)
    preprocessed_ds.set_format(type="torch", columns=["input_ids", "attention_mask", "labels", "target_length", "prompt_length"])
    def collate_fn(batch):
        max_len = max(len(item["input_ids"]) for item in batch)
        input_ids = []
        attention_mask = []
        labels = []
        target_lengths = []
        prompt_lengths = []
        for item in batch:
            input_ids.append(item["input_ids"])
            attention_mask.append(item["attention_mask"])
            labels.append(item["labels"])
            target_lengths.append(item["target_length"])
            prompt_lengths.append(item["prompt_length"])
        
        # Pad sequences
        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
        attention_mask = pad_sequence(attention_mask, batch_first=True, padding_value=0)
        labels = pad_sequence(labels, batch_first=True, padding_value=-100)
        target_lengths = torch.tensor(target_lengths)
        prompt_lengths = torch.tensor(prompt_lengths)
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "target_length": target_lengths,
            "prompt_length": prompt_lengths
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
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--dataset_name", type=str, default="HuggingFaceH4/ultrafeedback_binarized")
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=2)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--seed", type=int, default=16)
    # Intervals for validation and generation
    parser.add_argument("--validation_interval", type=int, default=1000, 
                        help="Run validation every N optimization steps (0 to disable)")
    parser.add_argument("--sample_generation_interval", type=int, default=200, 
                        help="Generate samples every N optimization steps (0 to disable)")
    # Length penalty parameters
    parser.add_argument("--length_penalty_interval", type=int, default=100,
                        help="Check generation lengths and apply penalty every N steps (0 to disable)")
    parser.add_argument("--length_penalty_factor", type=float, default=0.001,
                        help="Scaling factor for length penalty")
    # Early stopping parameters
    parser.add_argument("--early_stopping_steps", type=int, default=2000,
                        help="Number of validation steps without improvement before early stopping")
    return parser.parse_args()
if __name__ == "__main__":
    main()