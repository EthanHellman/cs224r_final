import json
import argparse
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm
from openai import OpenAI
import os

def load_ultrafeedback_data(num_samples=50):
    """Load the last N examples from train_prefs split"""
    print(f"Loading UltraFeedback dataset...")
    ds = load_dataset("HuggingFaceH4/ultrafeedback_binarized", split="train_prefs")
    
    # Get the last N samples
    total_samples = len(ds)
    start_idx = max(0, total_samples - num_samples)
    ds = ds.select(range(start_idx, total_samples))
    
    # Extract prompts, chosen and rejected responses
    data = []
    for example in ds:
        data.append({
            "prompt": example["prompt"],
            "chosen": example["chosen"][1]["content"],  # Assistant's response
            "rejected": example["rejected"][1]["content"]  # Assistant's response
        })
    
    return data

def format_prompts_for_generation(prompts, tokenizer):
    """Format prompts using chat template"""
    formatted_prompts = []
    for prompt in prompts:
        messages = [{"role": "user", "content": prompt}]
        formatted = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        formatted_prompts.append(formatted)
    return formatted_prompts

def generate_responses(model_path, prompts, model_name="model"):
    """Generate responses using VLLM"""
    print(f"\nGenerating responses from {model_name}...")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Format prompts
    formatted_prompts = format_prompts_for_generation(prompts, tokenizer)
    
    # Initialize VLLM
    llm = LLM(
        model=model_path,
        tensor_parallel_size=1,
        max_model_len=2048,
        dtype="float16",
        gpu_memory_utilization=0.8
    )
    
    # Set sampling parameters
    sampling_params = SamplingParams(
        temperature=0.7,
        top_p=0.9,
        max_tokens=512,
    )
    
    # Generate
    outputs = llm.generate(formatted_prompts, sampling_params)
    
    # Extract responses
    responses = []
    for output in outputs:
        responses.append(output.outputs[0].text)
    
    # Clean up to free GPU memory
    del llm
    import torch
    torch.cuda.empty_cache()
    
    return responses

def score_with_nemotron(prompt, response, client):
    """Score a single prompt-response pair using Nemotron"""
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
        # The score is typically in format "Score: X.XX"
        if score_text:
            score = float(score_text[7:])
        else:
            # Try to extract any float from the response
            import re
            numbers = re.findall(r'-?\d+\.?\d*', score_text)
            score = float(numbers[0]) if numbers else None
        
        return score
    except Exception as e:
        print(f"Error scoring with Nemotron: {e}")
        return None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_samples", type=int, default=50)
    parser.add_argument("--output_file", type=str, default="model_comparison_results_2.json")
    parser.add_argument("--api_key", type=str, required=True, help="NVIDIA API key for Nemotron")
    args = parser.parse_args()
    
    # Define model paths
    model_configs = [
        {
            "path": "/home/ec2-user/cs224r_final/checkpoints/sft_final_early_stop_step_5000",
            "name": "SFT",
            "key": "sft_response"
        },
        {
            "path": "/home/ec2-user/cs224r_final/checkpoints/dpo_best_epoch_2",
            "name": "DPO",
            "key": "dpo_response"
        },
        {
            "path": "/home/ec2-user/cs224r_final/checkpoints/rloo_best_epoch_0",
            "name": "RLOO",
            "key": "rloo_response"
        },
        {
            "path": "Qwen/Qwen2.5-0.5B-Instruct",
            "name": "Base Qwen2.5-0.5B-Instruct",
            "key": "base_response"
        }
    ]
    
    # Load data
    data = load_ultrafeedback_data(args.num_samples)
    prompts = [item["prompt"] for item in data]
    
    # Generate responses from each model
    for config in model_configs:
        responses = generate_responses(config["path"], prompts, config["name"])
        for i, response in enumerate(responses):
            data[i][config["key"]] = response
    
    # Initialize Nemotron client
    print("\nInitializing Nemotron client...")
    client = OpenAI(
        base_url="https://integrate.api.nvidia.com/v1",
        api_key=args.api_key
    )
    
    # Score all responses
    print("\nScoring all responses with Nemotron...")
    for i, item in enumerate(tqdm(data)):
        prompt = item["prompt"]
        
        # Score chosen response
        item["chosen_score"] = score_with_nemotron(prompt, item["chosen"], client)
        
        # Score rejected response
        item["rejected_score"] = score_with_nemotron(prompt, item["rejected"], client)
        
        # Score generated responses
        item["sft_score"] = score_with_nemotron(prompt, item["sft_response"], client)
        item["dpo_score"] = score_with_nemotron(prompt, item["dpo_response"], client)
        item["rloo_score"] = score_with_nemotron(prompt, item["rloo_response"], client)
        item["base_score"] = score_with_nemotron(prompt, item["base_response"], client)
        
        # Print progress every 10 samples
        if (i + 1) % 10 == 0:
            print(f"Processed {i + 1}/{len(data)} samples")
    
    # Save results
    print(f"\nSaving results to {args.output_file}")
    with open(args.output_file, 'w') as f:
        json.dump(data, f, indent=2)
    
    # Print summary statistics
    print("\nSummary Statistics:")
    print("-" * 50)
    
    # Calculate average scores
    score_keys = ["chosen_score", "rejected_score", "sft_score", "dpo_score", "rloo_score", "base_score"]
    for key in score_keys:
        scores = [item[key] for item in data if item[key] is not None]
        if scores:
            avg_score = sum(scores) / len(scores)
            print(f"Average {key}: {avg_score:.3f}")
    
    # Print a sample result
    print("\nSample result (first item):")
    print("-" * 50)
    sample = data[0]
    print(f"Prompt: {sample['prompt'][:100]}...")
    print(f"\nChosen response score: {sample['chosen_score']}")
    print(f"Rejected response score: {sample['rejected_score']}")
    print(f"SFT response score: {sample['sft_score']}")
    print(f"DPO response score: {sample['dpo_score']}")
    print(f"RLOO response score: {sample['rloo_score']}")
    print(f"Base response score: {sample['base_score']}")
    
    print(f"\nResults saved to {args.output_file}")

if __name__ == "__main__":
    main()