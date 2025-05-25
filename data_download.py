from datasets import Dataset
from datasets import load_dataset_builder
from datasets import get_dataset_split_names
from datasets import load_dataset

from transformers import AutoTokenizer
from transformers import DataCollatorWithPadding

import torch

def main():
    print(torch.cuda.is_available())
    print(torch.cuda.device_count())

    ds_builder = load_dataset_builder("HuggingFaceH4/ultrafeedback_binarized")

    ds = load_dataset("HuggingFaceH4/ultrafeedback_binarized", split="train_sft")

    ds = ds.select(range(50))

    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")

    # print(tokenizer.chat_template)

    # def tokenize(e):
    #     tokenized = tokenizer.apply_chat_template(
    #         e["chosen"],
    #         tokenize=True,
    #         add_generation_prompt=False,
    #         return_tensors=None,
    #         truncation=True,
    #         padding=False
    #     )

    #     return {"input_ids": tokenized}

    
    
    # ds = ds.map(tokenize)
    # print(ds[0])

    # ds.set_format(type="python", columns=["prompt", "prompt_id", "chosen", "input_ids"])

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer, return_tensors="pt")

    dataloader = torch.utils.data.DataLoader(
        ds, 
        batch_size=16, 
        collate_fn=lambda batch: data_collator(
            [
                tokenizer(
                    tokenizer.apply_chat_template(
                        e["chosen"],
                        tokenize=False,
                        add_generation_prompt=False
                    ),
                    return_tensors=None,
                    truncation=True,
                    padding=False
                ) for e in batch
            ]
        )
    )

    print("_" * 50)
    print(next(iter(dataloader)))


if __name__ == "__main__":
    main()