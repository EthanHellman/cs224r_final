import json
import argparse
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from tqdm import tqdm

def load_test_prompts(file_path):
    """Load test prompts from the ultrafeedback.jsonl file"""
    prompts = []
    with open(file_path, 'r') as f:
        for line in f:
            data = json.loads(line)
            if data.get("prompt"):
                prompts.append(data["prompt"])
    return prompts

def format_prompts(prompts, tokenizer):
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

def main():
    parser = argparse.ArgumentParser()
    # parser.add_argument("--checkpoint_path", type=str, default="./checkpoints/sft_epoch_10")
    parser.add_argument("--checkpoint_path", type=str, default="./checkpoints/rloo_best_epoch_0")
    parser.add_argument("--test_file", type=str, default="ultrafeedback.jsonl")
    parser.add_argument("--output_file", type=str, default="ultrafeedback_submission_vllm_rloo.json")
    parser.add_argument("--max_tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    parser.add_argument("--num_samples", type=int, default=None)
    args = parser.parse_args()
    
    # Load test prompts
    print(f"Loading test prompts from {args.test_file}")
    test_prompts = load_test_prompts(args.test_file)
    
    # Limit number of samples if specified
    if args.num_samples:
        test_prompts = test_prompts[:args.num_samples]
    
    print(f"Loaded {len(test_prompts)} prompts")
    
    # Load tokenizer for formatting
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Format prompts
    print("Formatting prompts...")
    formatted_prompts = format_prompts(test_prompts, tokenizer)
    
    # Initialize VLLM
    print(f"Initializing VLLM with model from {args.checkpoint_path}")
    # llm = LLM(
    #     model=args.checkpoint_path,
    #     tensor_parallel_size=args.tensor_parallel_size,
    #     max_model_len=1280,  # 256 prompt + 1024 response
    #     dtype="float16"
    # )

    llm = LLM(
        model=args.checkpoint_path,
        tensor_parallel_size=args.tensor_parallel_size,
        max_model_len=1280,  # 256 prompt + 1024 response
        dtype="float16",
        disable_custom_all_reduce=True,  # This can help with some issues
        enforce_eager=True  # This disables torch.compile
    )
    
    # Set sampling parameters
    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
    )
    
    # Generate responses
    print("Generating responses...")
    outputs = llm.generate(formatted_prompts, sampling_params)
    
    # Process outputs
    results = []
    for i, output in enumerate(outputs):
        # Extract only the generated text (not the prompt)
        generated_text = output.outputs[0].text
        
        results.append({
            "prompt": test_prompts[i],
            "response": generated_text
        })
    
    # Save to file
    print(f"Saving {len(results)} responses to {args.output_file}")
    with open(args.output_file, 'w') as f:
        for result in results:
            f.write(json.dumps(result) + '\n')
    
    print("Done!")
    
    # Print a few examples
    print("\nFirst 3 examples:")
    for i, result in enumerate(results[:3]):
        print(f"\nExample {i+1}:")
        print(f"Prompt: {result['prompt'][:100]}...")
        print(f"Response: {result['response'][:200]}...")

if __name__ == "__main__":
    main()