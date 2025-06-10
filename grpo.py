import argparse
import json
import os
import random
import re
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import wandb
import numpy as np
from tqdm import tqdm
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
    get_linear_schedule_with_warmup
)
from openai import OpenAI
from datasets import load_dataset

def parse_args():
    parser = argparse.ArgumentParser(description="Simplified GRPO Implementation")
    parser.add_argument("--model_path", type=str, default="./checkpoints/final",
                        help="Path to the model checkpoint")
    parser.add_argument("--dataset_name", type=str, default="HuggingFaceH4/ultrafeedback_binarized",
                        help="Dataset to use for training")
    parser.add_argument("--num_prompts", type=int, default=50,
                        help="Number of prompts to sample")
    parser.add_argument("--responses_per_prompt", type=int, default=4,
                        help="Number of responses to generate per prompt")
    parser.add_argument("--learning_rate", type=float, default=5e-6,
                        help="Learning rate for training")
    parser.add_argument("--kl_coef", type=float, default=0.1,
                        help="KL divergence coefficient")
    parser.add_argument("--max_steps", type=int, default=1000,
                        help="Maximum number of training steps")
    parser.add_argument("--batch_size", type=int, default=2,
                        help="Batch size for training")
    parser.add_argument("--nemotron_api_key", type=str, 
                        default="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",
                        help="API key for Nemotron model")
    parser.add_argument("--max_length", type=int, default=256,
                        help="Maximum length for generated responses")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--output_dir", type=str, default="./grpo_outputs",
                        help="Directory to save outputs")
    parser.add_argument("--log_interval", type=int, default=5,
                        help="Interval for logging")
    parser.add_argument("--evaluation_interval", type=int, default=20,
                        help="Interval for evaluation")
    parser.add_argument("--save_interval", type=int, default=50,
                        help="Interval for saving checkpoints")
    parser.add_argument("--use_wandb", action="store_true",
                        help="Whether to use wandb for logging")
    parser.add_argument("--debug", action="store_true",
                        help="Enable debug mode")
    return parser.parse_args()

def clean_response(response_text):
    """
    Extract only the assistant's response from the model output
    """
    try:
        # Handle case where response might be incomplete or malformed
        if '<|im_start|>assistant' not in response_text:
            return response_text.strip()
        
        # Split on assistant tag and take everything after it
        parts = response_text.split('<|im_start|>assistant', 1)
        if len(parts) < 2:
            return response_text.strip()
        
        # Remove the newline that typically follows the assistant tag
        assistant_content = parts[1]
        if assistant_content.startswith('\n'):
            assistant_content = assistant_content[1:]
        
        # Split on end tag and take everything before it
        if '<|im_end|>' in assistant_content:
            assistant_content = assistant_content.split('<|im_end|>', 1)[0]
        
        # Remove any endoftext tokens
        assistant_content = assistant_content.replace('<|endoftext|>', '')
        
        return assistant_content.strip()
    except Exception as e:
        print(f"Error cleaning response: {e}")
        return response_text.strip()

def sample_prompts(dataset_name, num_prompts, seed=42):
    """
    Sample diverse prompts from the dataset
    """
    print(f"Sampling {num_prompts} prompts from {dataset_name}...")
    
    # Set random seed for reproducibility
    random.seed(seed)
    
    # Load dataset
    ds = load_dataset(dataset_name, split="train_prefs")
    
    # Sample random indices
    indices = random.sample(range(len(ds)), min(num_prompts * 2, len(ds)))
    
    # Get prompts
    sampled_data = []
    prompts_set = set()  # To ensure unique prompts
    
    for idx in indices:
        prompt = ds[idx]["prompt"]
        
        # Only add if we don't have this prompt yet and not too long
        if prompt not in prompts_set and len(prompt.split()) < 200:
            prompts_set.add(prompt)
            sampled_data.append({
                "prompt": prompt,
                "original_chosen": ds[idx]["chosen"][1]["content"],
                "original_rejected": ds[idx]["rejected"][1]["content"]
            })
            
            # Break if we have enough prompts
            if len(sampled_data) >= num_prompts:
                break
    
    print(f"Sampled {len(sampled_data)} unique prompts")
    return sampled_data

def generate_responses(model, tokenizer, prompts_data, responses_per_prompt, max_length, device):
    """
    Generate multiple responses for each prompt with different parameters
    """
    print("Generating responses...")
    model.eval()
    all_responses = []
    
    # Define a range of temperature values to sample from
    temperature_values = [0.7, 0.8, 0.9, 1.0, 1.1]
    top_p_values = [0.92, 0.95, 0.98, 1.0]
    
    for prompt_data in tqdm(prompts_data):
        prompt = prompt_data["prompt"]
        
        # Format prompt using the chat template
        formatted_prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True
        )
        
        # Tokenize the prompt
        inputs = tokenizer(formatted_prompt, return_tensors="pt").to(device)
        input_length = inputs.input_ids.shape[1]
        
        prompt_responses = []
        
        # Generate multiple responses with different parameters
        for _ in range(responses_per_prompt):
            # Randomly select generation parameters
            temperature = random.choice(temperature_values)
            top_p = random.choice(top_p_values)
            
            # Generate response
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_length,
                    do_sample=True,
                    temperature=temperature,
                    top_p=top_p,
                    pad_token_id=tokenizer.pad_token_id
                )
            
            # Extract generated text (excluding the prompt)
            generated_tokens = outputs[0][input_length:]
            generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
            
            # Clean the response
            cleaned_text = clean_response(generated_text)
            
            # Record response with generation parameters
            prompt_responses.append({
                "response": cleaned_text,
                "temperature": temperature,
                "top_p": top_p
            })
        
        # Add original chosen response from the dataset as an additional response
        prompt_responses.append({
            "response": prompt_data["original_chosen"],
            "is_original": True,
            "temperature": 0,  # Placeholder values
            "top_p": 0
        })
        
        # Collect all responses for this prompt
        all_responses.append({
            "prompt": prompt,
            "responses": prompt_responses
        })
    
    return all_responses

def score_responses_with_nemotron(prompts_with_responses, api_key):
    """
    Score responses using the Nemotron reward model
    """
    print("Scoring responses with Nemotron...")
    
    # Initialize OpenAI client for Nemotron API
    client = OpenAI(
        base_url="https://integrate.api.nvidia.com/v1",
        api_key=api_key
    )
    
    scored_data = []
    
    for prompt_data in tqdm(prompts_with_responses):
        prompt = prompt_data["prompt"]
        scored_responses = []
        
        for response_data in prompt_data["responses"]:
            response = response_data["response"]
            
            # Score with Nemotron
            try:
                completion = client.chat.completions.create(
                    model="nvidia/llama-3.1-nemotron-70b-reward",
                    messages=[
                        {"role": "user", "content": prompt},
                        {"role": "assistant", "content": response}
                    ],
                )
                
                # Extract score from completion
                score_text = completion.choices[0].message.content
                # The score is usually returned as "score: X.XX"
                # match = re.search(r'score: ([-+]?\d*\.\d+|\d+)', score_text)
                # if match:
                #     score = float(match.group(1))
                # else:
                score = float(score_text[7:])  # Alternative parsing
                
            except Exception as e:
                print(f"Error scoring with Nemotron: {e}")
                score = 0.0
            
            # Add score to response data
            response_with_score = response_data.copy()
            response_with_score["score"] = score
            scored_responses.append(response_with_score)
            
            # Print some examples to verify
            if len(scored_data) < 2 and len(scored_responses) <= 2:
                print(f"\nExample prompt: {prompt[:100]}...")
                print(f"Response: {response[:100]}...")
                print(f"Score: {score}")
        
        # Sort responses by score (highest first)
        scored_responses.sort(key=lambda x: x["score"], reverse=True)
        
        # Add to scored data
        scored_data.append({
            "prompt": prompt,
            "responses": scored_responses
        })
    
    return scored_data

def prepare_training_data(scored_data, tokenizer, device):
    """
    Prepare data for GRPO training
    """
    print("Preparing training data...")
    
    training_data = []
    
    for prompt_data in scored_data:
        prompt = prompt_data["prompt"]
        responses = prompt_data["responses"]
        
        # Skip if fewer than 2 responses
        if len(responses) < 2:
            continue
        
        # Use the highest and lowest scored responses for training
        best_response = responses[0]["response"]
        worst_response = responses[-1]["response"]
        
        # Format the prompt and responses using the chat template
        formatted_prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True
        )
        
        best_formatted = tokenizer.apply_chat_template(
            [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": best_response}
            ],
            tokenize=False,
            add_generation_prompt=False
        )
        
        worst_formatted = tokenizer.apply_chat_template(
            [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": worst_response}
            ],
            tokenize=False,
            add_generation_prompt=False
        )
        
        # Tokenize
        prompt_ids = tokenizer(formatted_prompt, return_tensors=None)["input_ids"]
        best_ids = tokenizer(best_formatted, return_tensors=None)["input_ids"]
        worst_ids = tokenizer(worst_formatted, return_tensors=None)["input_ids"]
        
        # Prepare labels (mask prompt tokens)
        best_labels = best_ids.copy()
        best_labels[:len(prompt_ids)] = [-100] * len(prompt_ids)
        
        worst_labels = worst_ids.copy()
        worst_labels[:len(prompt_ids)] = [-100] * len(prompt_ids)
        
        # Store everything
        training_data.append({
            "prompt": prompt,
            "best_response": best_response,
            "worst_response": worst_response,
            "best_score": responses[0]["score"],
            "worst_score": responses[-1]["score"],
            "best_input_ids": best_ids,
            "best_labels": best_labels,
            "worst_input_ids": worst_ids,
            "worst_labels": worst_labels,
            "prompt_ids": prompt_ids
        })
    
    return training_data

def collate_batch(batch, tokenizer):
    """
    Collate a batch of training examples
    """
    max_len = max(
        max(len(item["best_input_ids"]) for item in batch),
        max(len(item["worst_input_ids"]) for item in batch)
    )
    
    best_input_ids = []
    best_attention_mask = []
    best_labels = []
    
    worst_input_ids = []
    worst_attention_mask = []
    worst_labels = []
    
    prompt_lengths = []
    
    for item in batch:
        # For best response
        b_input_ids = item["best_input_ids"]
        b_attn_mask = [1] * len(b_input_ids)
        b_labels = item["best_labels"]
        
        # For worst response
        w_input_ids = item["worst_input_ids"]
        w_attn_mask = [1] * len(w_input_ids)
        w_labels = item["worst_labels"]
        
        # Pad sequences
        if len(b_input_ids) < max_len:
            b_input_ids = b_input_ids + [tokenizer.pad_token_id] * (max_len - len(b_input_ids))
            b_attn_mask = b_attn_mask + [0] * (max_len - len(b_attn_mask))
            b_labels = b_labels + [-100] * (max_len - len(b_labels))
        
        if len(w_input_ids) < max_len:
            w_input_ids = w_input_ids + [tokenizer.pad_token_id] * (max_len - len(w_input_ids))
            w_attn_mask = w_attn_mask + [0] * (max_len - len(w_attn_mask))
            w_labels = w_labels + [-100] * (max_len - len(w_labels))
        
        best_input_ids.append(torch.tensor(b_input_ids))
        best_attention_mask.append(torch.tensor(b_attn_mask))
        best_labels.append(torch.tensor(b_labels))
        
        worst_input_ids.append(torch.tensor(w_input_ids))
        worst_attention_mask.append(torch.tensor(w_attn_mask))
        worst_labels.append(torch.tensor(w_labels))
        
        prompt_lengths.append(len(item["prompt_ids"]))
    
    return {
        "best_input_ids": torch.stack(best_input_ids),
        "best_attention_mask": torch.stack(best_attention_mask),
        "best_labels": torch.stack(best_labels),
        "worst_input_ids": torch.stack(worst_input_ids),
        "worst_attention_mask": torch.stack(worst_attention_mask),
        "worst_labels": torch.stack(worst_labels),
        "prompt_lengths": torch.tensor(prompt_lengths)
    }

def get_logprobs(logits, labels, attention_mask=None):
    """
    Calculate log probabilities from logits and labels
    """
    # Shift so that tokens < n predict n
    logits = logits[:, :-1, :].contiguous()
    labels = labels[:, 1:].contiguous()
    
    if attention_mask is not None:
        attention_mask = attention_mask[:, 1:].contiguous()
    
    # Get log probs
    log_probs = F.log_softmax(logits, dim=-1)
    
    # Create mask for valid tokens (not -100)
    mask = (labels != -100)
    
    if attention_mask is not None:
        mask = mask & (attention_mask == 1)
    
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

def compute_grpo_loss(model, ref_model, batch, kl_coef, device):
    """
    Compute the GRPO loss
    """
    # Move batch to device
    batch = {k: v.to(device) for k, v in batch.items()}
    
    # Process best responses
    best_outputs = model(
        input_ids=batch["best_input_ids"],
        attention_mask=batch["best_attention_mask"]
    )
    
    # Process worst responses
    worst_outputs = model(
        input_ids=batch["worst_input_ids"],
        attention_mask=batch["worst_attention_mask"]
    )
    
    # Get reference model outputs
    with torch.no_grad():
        ref_best_outputs = ref_model(
            input_ids=batch["best_input_ids"],
            attention_mask=batch["best_attention_mask"]
        )
        
        ref_worst_outputs = ref_model(
            input_ids=batch["worst_input_ids"],
            attention_mask=batch["worst_attention_mask"]
        )
    
    # Calculate log probs
    best_logprobs = get_logprobs(best_outputs.logits, batch["best_labels"], batch["best_attention_mask"])
    worst_logprobs = get_logprobs(worst_outputs.logits, batch["worst_labels"], batch["worst_attention_mask"])
    
    ref_best_logprobs = get_logprobs(ref_best_outputs.logits, batch["best_labels"], batch["best_attention_mask"])
    ref_worst_logprobs = get_logprobs(ref_worst_outputs.logits, batch["worst_labels"], batch["worst_attention_mask"])
    
    # Calculate KL divergence from reference model
    best_kl = best_logprobs - ref_best_logprobs
    worst_kl = worst_logprobs - ref_worst_logprobs
    
    # GRPO loss = -(best_logprobs - worst_logprobs) + kl_coef * (best_kl + worst_kl)
    reward_diff = best_logprobs - worst_logprobs
    kl_penalty = kl_coef * (best_kl + worst_kl)
    
    # Final loss (negative because we want to maximize reward difference)
    loss = -reward_diff.mean() + kl_penalty.mean()
    
    # Calculate accuracy (best > worst)
    accuracy = (best_logprobs > worst_logprobs).float().mean()
    
    return loss, accuracy, reward_diff.mean(), kl_penalty.mean()

def train_grpo(model, ref_model, training_data, tokenizer, args, device):
    """
    Train the model using GRPO
    """
    print("Starting GRPO training...")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize wandb
    if args.use_wandb:
        wandb.init(
            project="simplified-grpo",
            config=vars(args),
            name=f"grpo_{os.path.basename(args.model_path)}"
        )
    
    # Set model to training mode
    model.train()
    ref_model.eval()  # Reference model always in eval mode
    
    # Create optimizer and scheduler
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    
    num_steps = args.max_steps
    num_warmup_steps = int(0.1 * num_steps)
    
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_steps
    )
    
    # Create dataloader
    batch_size = args.batch_size
    train_indices = list(range(len(training_data)))
    
    # Main training loop
    best_accuracy = 0
    model.gradient_checkpointing_enable()
    
    for step in range(args.max_steps):
        # Sample a batch
        batch_indices = random.sample(train_indices, min(batch_size, len(train_indices)))
        batch_data = [training_data[i] for i in batch_indices]
        
        # Collate batch
        batch = collate_batch(batch_data, tokenizer)
        
        # Forward pass and compute loss
        loss, accuracy, reward_diff, kl_penalty = compute_grpo_loss(
            model, ref_model, batch, args.kl_coef, device
        )
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        
        # Clip gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        # Update parameters
        optimizer.step()
        scheduler.step()
        
        # Log metrics
        if step % args.log_interval == 0:
            curr_lr = scheduler.get_last_lr()[0]
            print(f"Step: {step}, Loss: {loss.item():.4f}, Accuracy: {accuracy.item():.4f}, "
                  f"Reward Diff: {reward_diff.item():.4f}, KL Penalty: {kl_penalty.item():.4f}, "
                  f"LR: {curr_lr:.2e}")
            
            if args.use_wandb:
                wandb.log({
                    "loss": loss.item(),
                    "accuracy": accuracy.item(),
                    "reward_diff": reward_diff.item(),
                    "kl_penalty": kl_penalty.item(),
                    "learning_rate": curr_lr,
                    "step": step
                })
        
        # Evaluate
        if (step + 1) % args.evaluation_interval == 0 or step == args.max_steps - 1:
            eval_accuracy = evaluate_grpo(model, training_data, tokenizer, device, args)
            
            print(f"Evaluation at step {step}: Accuracy = {eval_accuracy:.4f}")
            
            if args.use_wandb:
                wandb.log({
                    "eval_accuracy": eval_accuracy,
                    "step": step
                })
            
            # Save best model
            if eval_accuracy > best_accuracy:
                best_accuracy = eval_accuracy
                save_checkpoint(model, tokenizer, os.path.join(args.output_dir, "best_model"))
                print(f"New best model saved with accuracy: {best_accuracy:.4f}")
        
        # Save checkpoint
        if (step + 1) % args.save_interval == 0:
            save_checkpoint(model, tokenizer, os.path.join(args.output_dir, f"checkpoint_step_{step+1}"))
    
    # Save final model
    save_checkpoint(model, tokenizer, os.path.join(args.output_dir, "final_model"))
    
    # Generate samples with the final model
    print("Generating samples with the final model...")
    generate_and_log_samples(model, tokenizer, training_data, device, args)
    
    # Finish wandb run
    if args.use_wandb:
        wandb.finish()
    
    return model

def evaluate_grpo(model, training_data, tokenizer, device, args):
    """
    Evaluate the model on the training data
    """
    model.eval()
    total_correct = 0
    total_samples = 0
    
    eval_batch_size = args.batch_size
    eval_indices = list(range(len(training_data)))
    
    with torch.no_grad():
        for i in range(0, len(eval_indices), eval_batch_size):
            batch_indices = eval_indices[i:i+eval_batch_size]
            batch_data = [training_data[j] for j in batch_indices]
            
            # Collate batch
            batch = collate_batch(batch_data, tokenizer)
            batch = {k: v.to(device) for k, v in batch.items()}
            
            # Process best responses
            best_outputs = model(
                input_ids=batch["best_input_ids"],
                attention_mask=batch["best_attention_mask"]
            )
            
            # Process worst responses
            worst_outputs = model(
                input_ids=batch["worst_input_ids"],
                attention_mask=batch["worst_attention_mask"]
            )
            
            # Calculate log probs
            best_logprobs = get_logprobs(best_outputs.logits, batch["best_labels"], batch["best_attention_mask"])
            worst_logprobs = get_logprobs(worst_outputs.logits, batch["worst_labels"], batch["worst_attention_mask"])
            
            # Count correct predictions (best > worst)
            correct = (best_logprobs > worst_logprobs).sum().item()
            
            total_correct += correct
            total_samples += len(batch_indices)
    
    accuracy = total_correct / total_samples
    model.train()
    return accuracy

def generate_and_log_samples(model, tokenizer, training_data, device, args, num_samples=5):
    """
    Generate samples from the model and log them
    """
    model.eval()
    samples = []
    
    # Select a subset of prompts
    sample_indices = random.sample(range(len(training_data)), min(num_samples, len(training_data)))
    
    for idx in sample_indices:
        data = training_data[idx]
        prompt = data["prompt"]
        
        # Format prompt
        formatted_prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True
        )
        
        # Tokenize
        inputs = tokenizer(formatted_prompt, return_tensors="pt").to(device)
        input_length = inputs.input_ids.shape[1]
        
        # Generate response
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=args.max_length,
                do_sample=True,
                temperature=0.9,
                top_p=0.95,
                pad_token_id=tokenizer.pad_token_id
            )
        
        # Extract generated text
        generated_tokens = outputs[0][input_length:]
        generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
        
        # Clean response
        cleaned_text = clean_response(generated_text)
        
        # Add to samples
        samples.append({
            "prompt": prompt,
            "generated": cleaned_text,
            "best": data["best_response"],
            "worst": data["worst_response"]
        })
    
    # Save samples to file
    samples_file = os.path.join(args.output_dir, "generated_samples.json")
    with open(samples_file, "w") as f:
        json.dump(samples, f, indent=2)
    
    # Log to wandb
    if args.use_wandb:
        wandb.log({
            "generated_samples": wandb.Table(
                columns=["prompt", "generated", "best", "worst"],
                data=[[s["prompt"], s["generated"], s["best"], s["worst"]] for s in samples]
            )
        })
    
    model.train()
    return samples

def save_checkpoint(model, tokenizer, path):
    """
    Save model and tokenizer checkpoint
    """
    model.save_pretrained(path)
    tokenizer.save_pretrained(path)
    print(f"Checkpoint saved to: {path}")

def main():
    # Parse command-line arguments
    args = parse_args()
    
    # Set random seed
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load tokenizer
    print(f"Loading tokenizer from {args.model_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load models
    print(f"Loading model from {args.model_path}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype="auto",
        device_map="auto"
    )
    
    print("Loading reference model")
    ref_model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype="auto",
        device_map="auto"
    )
    
    # Freeze reference model
    for param in ref_model.parameters():
        param.requires_grad = False
    
    # Sample prompts
    prompts_data = sample_prompts(args.dataset_name, args.num_prompts, args.seed)
    
    # Generate responses
    responses_data = generate_responses(
        model, tokenizer, prompts_data, 
        args.responses_per_prompt, args.max_length, device
    )
    
    # Score responses
    scored_data = score_responses_with_nemotron(responses_data, args.nemotron_api_key)
    
    # Save scored data
    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "scored_data.json"), "w") as f:
        json.dump(scored_data, f, indent=2)
    
    # Prepare training data
    training_data = prepare_training_data(scored_data, tokenizer, device)
    
    # Save training data
    with open(os.path.join(args.output_dir, "training_data.json"), "w") as f:
        training_data_json = []
        for item in training_data:
            item_copy = item.copy()
            # Remove token IDs to make the JSON readable
            for key in ['best_input_ids', 'best_labels', 'worst_input_ids', 'worst_labels', 'prompt_ids']:
                if key in item_copy:
                    del item_copy[key]
            training_data_json.append(item_copy)
        json.dump(training_data_json, f, indent=2)
    
    # Train model
    trained_model = train_grpo(model, ref_model, training_data, tokenizer, args, device)
    
    print("GRPO training complete!")
    return trained_model

if __name__ == "__main__":
    main()