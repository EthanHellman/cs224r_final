import json
import argparse
from openai import OpenAI
from tqdm import tqdm

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

def rescore_pregenerated_responses(input_file, api_key, output_file):
    """Re-score pre-generated responses using Nemotron reward model"""
    # Initialize OpenAI client for Nemotron
    client = OpenAI(
        base_url="https://integrate.api.nvidia.com/v1",
        api_key=api_key
    )
    
    # Load the pregenerated responses
    with open(input_file, 'r') as f:
        data = json.load(f)
    
    win_count = 0
    total_count = 0
    results = []
    
    print("Re-scoring responses with Nemotron...")
    for result in tqdm(data["detailed_results"], total=len(data["detailed_results"])):
        prompt = result["prompt"]
        trained_response = result["trained_response"]
        reference_response = result["reference_response"]
        
        # Score both responses with Nemotron
        trained_score = score_with_nemotron(prompt, trained_response, client)
        reference_score = score_with_nemotron(prompt, reference_response, client)
        
        if trained_score is not None and reference_score is not None:
            # Calculate win
            win = 1 if trained_score > reference_score else 0
            win_count += win
            total_count += 1
            
            results.append({
                "prompt": prompt,
                "trained_response": trained_response,
                "reference_response": reference_response,
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
    
    # Prepare output in the same format as the input file
    output = {
        "config": data.get("config", {}),
        "win_rate": win_rate,
        "total_samples": len(results),
        "detailed_results": results
    }
    
    # Save the results
    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\n{'='*50}")
    print(f"FINAL RESULTS")
    print(f"{'='*50}")
    print(f"Total samples evaluated: {len(results)}")
    print(f"Win rate: {win_rate:.2%}")
    print(f"{'='*50}")
    print(f"\nDetailed results saved to {output_file}")
    
    # Print some example comparisons
    print("\nExample comparisons:")
    for i, result in enumerate(results[:3]):
        print(f"\n--- Example {i+1} ---")
        print(f"Prompt: {result['prompt'][:100]}...")
        print(f"Trained score: {result['trained_score']:.2f}")
        print(f"Reference score: {result['reference_score']:.2f}")
        print(f"Winner: {'Trained' if result['win'] else 'Reference'}")
    
    return win_rate, results

def main():
    parser = argparse.ArgumentParser(description='Re-score pre-generated responses with Nemotron')
    parser.add_argument("--input_file", type=str, 
                        default="evaluation_results_cleaned.json",
                        help="Path to the input file with pregenerated responses")
    parser.add_argument("--api_key", type=str, default="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",
                        help="NVIDIA API key for Nemotron")
    parser.add_argument("--output_file", type=str, 
                        default="rescored_evaluation_results.json",
                        help="Path to save the new evaluation results")
    args = parser.parse_args()
    
    # Re-score the pregenerated responses
    rescore_pregenerated_responses(
        input_file=args.input_file,
        api_key=args.api_key,
        output_file=args.output_file
    )

if __name__ == "__main__":
    main()