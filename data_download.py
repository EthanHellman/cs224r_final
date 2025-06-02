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
            "model": args.model_name,
            "dataset": args.dataset_name,
            "epochs": args.num_epochs,
            "batch_size": args.batch_size,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "seed": args.seed
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
    # test_dataloader = get_dataloader(args, "test")

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype="auto",
        device_map="auto"
    )

    ## TESTING VALIDATION:
    # print("testing validation")
    # val_loss = validate(model, device, validation_dataloader, run)
    # print(val_loss)

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

    save_checkpoint(model, tokenizer, epoch=args.num_epochs)

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
    
    for epoch in range(args.num_epochs):
        model.train()
        epoch_loss = 0
        optimizer.zero_grad()

        for i, batch in enumerate(train_dataloader):
            batch = {k: v.to(device) for k, v in batch.items()}

            with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
                outputs = model(**batch)
                loss = outputs.loss / gradient_accumulation_steps
            
            loss.backward()

            # print(f"Step: {i}, Loss: {loss.item():.4f}")

            if i % 10 == 0:
                curr_lr = scheduler.get_last_lr()[0]
                actual_loss = loss.item() * gradient_accumulation_steps

                print(f"Epoch: {epoch}, Batch: {i}, Loss: {actual_loss:.4f}, Current LR: {curr_lr:.2e}")

                run.log({
                    "loss": actual_loss, 
                    "learning_rate": curr_lr,
                    # "gradient_norm": total_norm
                })


            if (i + 1) % gradient_accumulation_steps == 0:
                # print("Accumulating")
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
            
            del outputs, loss
            if i % 50 == 0:
                torch.cuda.empty_cache()

        if len(train_dataloader) % gradient_accumulation_steps != 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            
        avg_loss = epoch_loss / len(train_dataloader)
        # run.log({"avg_loss": avg_loss})

        train_losses.append(avg_loss)
        print(f"Epoch: {epoch}, Average Loss: {avg_loss}")

        val_loss, perplexity = validate(model, device, validation_dataloader)

        run.log({
            "avg_train_loss": avg_loss,
            "val_loss": val_loss,
            "val_perplexity": perplexity
        })

        print(f"Epoch: {epoch}, Validation Loss: {val_loss}, Validation Perplexity: {perplexity}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            save_checkpoint(model, tokenizer, epoch=f"best_epoch_{epoch}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch}")
                break
    
    if patience_counter > 0:
        print(f"Best model saved at checkpoint: best_epoch_{epoch - patience_counter}")
    else:
        print(f"Best model saved at checkpoint: best_epoch_{epoch}")

    return train_losses

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
    # run.log({
    #     "val_loss": val_loss,
    #     "val_perplexity": perplexity
    # })

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
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id
            )
        
        # print("length of outputs: ", len(outputs))
        generated_tokens = outputs[0][input_length:]
        generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
        # print(generated_text)

        samples.append({
            "prompt": prompt["prompt"],
            "response": generated_text,
            "chosen": prompt["chosen"]
        })
    
    return samples

def evaluate_feedback(args, model, tokenizer, device, run, num_samples=50):
    ds = load_dataset(args.dataset_name, split="test_sft")
    ds = ds.select(range(0, num_samples))

    # chosens = [example["chosen"][1]["content"] for example in ds]
    # prompts = [example["prompt"] for example in ds]

    prompts = [{"prompt": example["prompt"], "chosen": example["chosen"][1]["content"]} for example in ds]

    generations = generate_samples(model, tokenizer, prompts, device)

    run.log({
        "samples": wandb.Table(
            columns=["prompt", "generation", "chosen"],
            data=[[e["prompt"], e["response"], e["chosen"]] for e in generations]
        )
    })

    return generations

def save_checkpoint(model, tokenizer, epoch, checkpoint_dir="./checkpoints"):
    os.makedirs(checkpoint_dir, exist_ok=True)

    checkpoint_path = os.path.join(checkpoint_dir, f"sft_epoch_{epoch}")
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
        prompt = tokenizer.apply_chat_template(
            [example["chosen"][0]],
            tokenize=False,
            add_generation_prompt=True
        )

        chosen = tokenizer.apply_chat_template(
            example["chosen"],
            tokenize=False,
            add_generation_prompt=False
        )

        prompt_ids = tokenizer(prompt, return_tensors=None, truncation=True, padding=False)["input_ids"]
        full = tokenizer(chosen, return_tensors=None, truncation=True, padding=False)

        input_ids = full["input_ids"]
        attention_mask = full["attention_mask"]
        labels = input_ids.copy()
        labels[:len(prompt_ids)] = [-100] * len(prompt_ids)

        # print(labels)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }

    preprocessed_ds = ds.map(preprocess)
    preprocessed_ds.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

    def collate_fn(batch):
        max_len = max(len(item["input_ids"]) for item in batch)

        input_ids = []
        attention_mask = []
        labels = []

        for item in batch:
            input_ids.append(item["input_ids"])
            attention_mask.append(item["attention_mask"])
            labels.append(item["labels"])
        
        # Pad sequences
        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
        attention_mask = pad_sequence(attention_mask, batch_first=True, padding_value=0)
        labels = pad_sequence(labels, batch_first=True, padding_value=-100)
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
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
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=2)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--seed", type=int, default=16)
    return parser.parse_args()

if __name__ == "__main__":
    main()