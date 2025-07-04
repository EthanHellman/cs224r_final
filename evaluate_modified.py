import json
import argparse
import re
from vllm import LLM, SamplingParams
from vllm.sampling_params import BeamSearchParams
from transformers import AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm
from openai import OpenAI
import numpy as np
import os
import wandb

def clean_response(response_text):
    """
    Extract only the assistant's response by:
    1. Splitting on '<|im_start|>assistant\n' and taking everything after it
    2. Splitting the result on '<|im_end|>' and taking everything before it
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

def load_test_prompts(dataset_name, num_samples=200):
    """Load test prompts from ultrafeedback dataset"""
    ds = load_dataset(dataset_name, split="train_prefs")
    # Select a subset for evaluation
    ds = ds.select(range(14000, 14000 + num_samples))
    
    prompts = []
    for example in ds:
        prompts.append(example["prompt"])
    
    return prompts

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

def generate_responses(model_path, prompts, formatted_prompts, max_tokens=256, chunk_size=20):
    """Generate responses using VLLM beam search with length penalty"""
    print(f"Loading model from {model_path}")
    
    # Initialize VLLM
    llm = LLM(
        model=model_path,
        tensor_parallel_size=1,
        max_model_len=2048,
        dtype="float16",
        max_num_seqs=chunk_size,
        gpu_memory_utilization=0.8
    )
    
    # Create beam search parameters with length penalty
    beam_params = BeamSearchParams(
        beam_width=4,  # Number of beams to track
        max_tokens=max_tokens,
        length_penalty=0.9,  # Values < 1.0 penalize longer sequences
        temperature=0.9  # Some temperature for diversity
    )
    
    # Generate responses in chunks
    responses = []
    print(f"Generating responses in chunks of {chunk_size}...")
    
    for chunk_start in range(0, len(prompts), chunk_size):
        chunk_end = min(chunk_start + chunk_size, len(prompts))
        chunk_prompts = prompts[chunk_start:chunk_end]
        chunk_formatted = formatted_prompts[chunk_start:chunk_end]
        
        print(f"Processing chunk {chunk_start//chunk_size + 1}/{(len(prompts) + chunk_size - 1)//chunk_size}")
        
        try:
            # Convert prompts to required format for beam search
            beam_search_prompts = []
            for i, prompt in enumerate(chunk_formatted):
                # Need to use TextPrompt format for beam_search
                beam_search_prompts.append({"prompt": prompt})
            
            # Use beam_search directly
            outputs = llm.beam_search(beam_search_prompts, beam_params)
            
            # Extract responses (use the top beam for each prompt)
            for i, output in enumerate(outputs):
                best_beam = output.sequences[0]  # First beam is the best one
                generated_text = best_beam.text
                
                # Clean the response to extract just the assistant's part
                cleaned_text = clean_response(generated_text)
                
                responses.append({
                    "prompt": chunk_prompts[i],
                    "response": cleaned_text
                })
        except Exception as e:
            print(f"Error processing chunk: {e}")
            # Add empty responses for failed prompts
            for prompt in chunk_prompts:
                responses.append({
                    "prompt": prompt,
                    "response": "[Generation failed]"
                })
    
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
        score = float(completion.choices[0].message.content[7:])
        return score
    except Exception as e:
        print(f"Error scoring with Nemotron: {e}")
        return None

def evaluate_models(trained_responses, reference_responses, api_key):
    """Evaluate and compare models using Nemotron reward model"""
    # Initialize OpenAI client for Nemotron
    client = OpenAI(
        base_url="https://integrate.api.nvidia.com/v1",
        api_key=api_key
    )
    
    win_count = 0
    total_count = 0
    results = []
    
    print("Scoring responses with Nemotron...")
    for trained, reference in tqdm(zip(trained_responses, reference_responses), 
                                   total=len(trained_responses)):
        # Ensure prompts match
        assert trained["prompt"] == reference["prompt"], "Prompts don't match!"
        
        prompt = trained["prompt"]
        
        # Score both responses
        trained_score = score_with_nemotron(prompt, trained["response"], client)
        reference_score = score_with_nemotron(prompt, reference["response"], client)
        
        if trained_score is not None and reference_score is not None:
            print(trained_score)
            print(reference_score)
            # Calculate win
            win = 1 if trained_score >= reference_score else 0
            win_count += win
            total_count += 1
            
            results.append({
                "prompt": prompt,
                "trained_response": trained["response"],
                "reference_response": reference["response"],
                "trained_score": trained_score,
                "reference_score": reference_score,
                "win": win
            })
            
            # Print progress
            if total_count % 10 == 0:
                current_winrate = win_count / total_count
                print(f"Progress: {total_count} samples, Current win rate: {current_winrate:.2%}")
    
    # Calculate final win rate
    win_rate = win_count / total_count if total_count > 0 else 0
    
    return win_rate, results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trained_model_path", type=str,
                        # default="./checkpoints/sft_final_early_stop_step_5000",
                        # default="./checkpoints/final",
                        # default="./checkpoints/sft_final",
                        default="./checkpoints/dpo_epoch_0",
                        help="Path to your trained model checkpoint")
    parser.add_argument("--reference_model", type=str, 
                        default="Qwen/Qwen2.5-0.5B-Instruct",
                        help="Reference model for comparison")
    parser.add_argument("--dataset_name", type=str, 
                        default="HuggingFaceH4/ultrafeedback_binarized")
    parser.add_argument("--num_samples", type=int, default=200,
                        help="Number of test samples to evaluate")
    parser.add_argument("--api_key", type=str, default="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",
                        help="NVIDIA API key for Nemotron")
    parser.add_argument("--output_file", type=str, default="evaluation_results_dpo_old_epoch_0.json")
    parser.add_argument("--use_wandb", action="store_true",
                        help="Log results to wandb")
    args = parser.parse_args()
    
    # Initialize wandb if requested
    if args.use_wandb:
        run = wandb.init(
            entity="ethanhellman",
            project="CS224R",
            name=f"eval_{os.path.basename(args.trained_model_path)}",
            config=vars(args)
        )
    
    # Load test prompts
    print(f"Loading {args.num_samples} test prompts...")
    prompts = load_test_prompts(args.dataset_name, args.num_samples)
    
    # Load tokenizer from trained model
    tokenizer = AutoTokenizer.from_pretrained(args.trained_model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Format prompts
    formatted_prompts = format_prompts_for_generation(prompts, tokenizer)
    
    # Generate responses from trained model
    print("\nGenerating responses from trained model...")
    trained_responses = generate_responses(
        args.trained_model_path, 
        prompts, 
        formatted_prompts
    )
    
    # Load reference model tokenizer and format prompts
    ref_tokenizer = AutoTokenizer.from_pretrained(args.reference_model)
    if ref_tokenizer.pad_token is None:
        ref_tokenizer.pad_token = ref_tokenizer.eos_token
    ref_formatted_prompts = format_prompts_for_generation(prompts, ref_tokenizer)
    
    # Generate responses from reference model
    print("\nGenerating responses from reference model...")
    reference_responses = generate_responses(
        args.reference_model,
        prompts,
        ref_formatted_prompts
    )
    
    # Evaluate models
    print("\nEvaluating models with Nemotron...")
    win_rate, results = evaluate_models(
        trained_responses, 
        reference_responses, 
        args.api_key
    )
    
    # Print results
    print(f"\n{'='*50}")
    print(f"FINAL RESULTS")
    print(f"{'='*50}")
    print(f"Total samples evaluated: {len(results)}")
    print(f"Win rate: {win_rate:.2%}")
    print(f"{'='*50}")
    
    # Save detailed results
    output = {
        "config": vars(args),
        "win_rate": win_rate,
        "total_samples": len(results),
        "detailed_results": results
    }
    
    with open(args.output_file, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nDetailed results saved to {args.output_file}")
    
    # Log to wandb if enabled
    if args.use_wandb:
        run.log({
            "win_rate": win_rate,
            "total_samples": len(results)
        })
        
        # Create a table with sample results
        table_data = []
        for i, result in enumerate(results[:10]):  # First 10 for visualization
            table_data.append([
                result["prompt"][:100] + "...",
                result["trained_response"][:200] + "...",
                result["reference_response"][:200] + "...",
                result["trained_score"],
                result["reference_score"],
                result["win"]
            ])
        
        table = wandb.Table(
            columns=["Prompt", "Trained Response", "Reference Response", 
                    "Trained Score", "Reference Score", "Win"],
            data=table_data
        )
        run.log({"evaluation_samples": table})
        run.finish()
    
    # Print some example comparisons
    print("\nExample comparisons:")
    for i, result in enumerate(results[:3]):
        print(f"\n--- Example {i+1} ---")
        print(f"Prompt: {result['prompt'][:100]}...")
        print(f"Trained score: {result['trained_score']:.2f}")
        print(f"Reference score: {result['reference_score']:.2f}")
        print(f"Winner: {'Trained' if result['win'] else 'Reference'}")

if __name__ == "__main__":
    main()