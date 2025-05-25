from datasets import Dataset
from datasets import load_dataset_builder
from datasets import get_dataset_split_names
from datasets import load_dataset

from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import DataCollatorForLanguageModeling

import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence

import wandb
import argparse
import json
import os

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

    train(args, model, device, train_dataloader, validation_dataloader, run)

    samples = evaluate_feedback(args, model, tokenizer, device, run)

    with open("sft_samples.json", "w") as f:
        json.dump(samples, f, indent=2)

    save_checkpoint(model, tokenizer, epoch=args.num_epochs)

    run.finish()

def train(args, model, device, train_dataloader, validation_dataloader, run):
    print("Training")
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    model.train()
    
    train_losses = []
    
    for epoch in range(args.num_epochs):
        model.train()

        epoch_loss = 0
        for i, batch in enumerate(train_dataloader):
            batch = {k: v.to(device) for k, v in batch.items()}
            optimizer.zero_grad()
            
            # Use autocast without GradScaler for bfloat16
            with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
                outputs = model(**batch)
                loss = outputs.loss
            
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            if i % 10 == 0:
                print(f"Epoch: {epoch}, Batch: {i}, Loss: {loss.item():.4f}")

            run.log({"loss": loss.item()})
            
            del outputs, loss
            if i % 50 == 0:
                torch.cuda.empty_cache()
            
        avg_loss = epoch_loss / len(train_dataloader)
        run.log({"avg_loss": avg_loss})

        train_losses.append(avg_loss)
        print(f"Epoch: {epoch}, Average Loss: {avg_loss}")

        val_loss = validate(model, device, validation_dataloader, run)

        print(f"Epoch: {epoch}, Validation Loss: {val_loss}")
    
    return train_losses

def validate(model, device, dataloader, run):
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for batch in dataloader:
            batch = {k: v.to(device) for k, v in batch.items()} 

            with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
                outputs = model(**batch)
                loss = outputs.loss

            total_loss += loss.item()

    val_loss = total_loss / len(dataloader)
    run.log({"val_loss": val_loss})
    return val_loss

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

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_length,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id
            )
        
        # print("length of outputs: ", len(outputs))

        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
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
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--seed", type=int, default=16)
    return parser.parse_args()

if __name__ == "__main__":
    main()