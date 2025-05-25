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

model_name = "Qwen/Qwen2.5-0.5B"
dataset_name = "HuggingFaceH4/ultrafeedback_binarized"

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"CUDA device count: {torch.cuda.device_count()}")

    dataloader = get_dataloader()

    # batch = next(iter(dataloader))
    # batch_on_device = {k: v.to(device) for k, v in batch.items()}

    # print("Batch keys:", batch_on_device.keys())
    # print("Shape of input_ids:", batch_on_device["input_ids"].shape)
    # print("Device of input_ids:", batch_on_device["input_ids"].device)

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map="auto"
    )

    train(model, device, dataloader)

def train(model, device, dataloader):
    print("Training")

    optimizer = optim.Adam(model.parameters(), lr=1e-5)
    model.train()
    
    criterion = nn.NLLLoss()

    train_losses = []

    for epoch in range(10):
        epoch_loss = 0

        for i, batch in enumerate(dataloader):
            batch = {k: v.to(device) for k, v in batch.items()}

            optimizer.zero_grad()

            outputs = model(**batch)
            loss = outputs.loss
            
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

            if i % 10 == 0:
                print(f"Epoch: {epoch}, Batch: {i}, Loss: {loss.item():.4f}")

            del outputs, loss
            torch.cuda.empty_cache()

        avg_loss = epoch_loss / len(dataloader)
        train_losses.append(avg_loss)

        print(f"Epoch: {epoch}, Average Loss: {avg_loss}")

    return train_losses

def get_dataloader():
    ds = load_dataset(dataset_name, split="train_sft")

    # ds = ds.select(range(50))

    # print(ds[0])

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    def preprocess(example):
        prompt = tokenizer.apply_chat_template(
            [example["chosen"][0]],
            tokenize=False,
            add_generation_prompt=False
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

    # print(preprocessed_ds[0[]])

    # data_collator = DataCollatorWithPadding(tokenizer=tokenizer, return_tensors="pt")
    # data_collator = DataCollatorForLanguageModeling(
    #     tokenizer=tokenizer,
    #     mlm=False,
    #     return_tensors="pt"
    # )

    # dataloader = torch.utils.data.DataLoader(
    #     ds, 
    #     batch_size=16, 
    #     collate_fn=lambda batch: data_collator(
    #         [
    #             tokenizer(
    #                 tokenizer.apply_chat_template(
    #                     e["chosen"],
    #                     tokenize=False,
    #                     add_generation_prompt=False
    #                 ),
    #                 return_tensors=None,
    #                 truncation=True,
    #                 padding=False
    #             ) for e in batch
    #         ]
    #     )
    # )

    def collate_fn(batch):
        max_len = max(len(item["input_ids"]) for item in batch)

        input_ids = []
        attention_mask = []
        labels = []

        # for item in batch:
        #     curr_len = len(item["input_ids"])

        #     padding_len = max_len - curr_len

        #     padded_input_ids = item["input_ids"] + [tokenizer.pad_token_id] * padding_len
        #     input_ids.append(padded_input_ids)

        #     padded_attention_mask = item["attention_mask"] + [0] * padding_len
        #     attention_mask.append(padded_attention_mask)

        #     padded_labels = item["labels"] + [-100] * padding_len
        #     labels.append(padded_labels)

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
        batch_size=1,
        collate_fn=collate_fn
    )

    print("_" * 50)

    return dataloader

if __name__ == "__main__":
    main()