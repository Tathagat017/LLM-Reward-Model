import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import pandas as pd
import json
from typing import List, Dict

def setup_model():
    """Setup the base model for text generation"""
    model_name = "microsoft/DialoGPT-medium"  # Non-gated model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    
    # Add padding token if it doesn't exist
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    return tokenizer, model

def generate_answers(prompts: List[str], tokenizer, model, num_answers: int = 4) -> Dict[str, List[str]]:
    """Generate multiple candidate answers for each prompt"""
    results = {}
    
    for prompt in prompts:
        print(f"Generating answers for: {prompt[:50]}...")
        
        # Encode the prompt
        inputs = tokenizer.encode(prompt, return_tensors='pt')
        
        # Generate multiple responses
        with torch.no_grad():
            outputs = model.generate(
                inputs,
                max_length=inputs.shape[1] + 100,  # Add 100 tokens to the prompt length
                num_return_sequences=num_answers,
                temperature=0.8,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
                no_repeat_ngram_size=2
            )
        
        # Decode the responses
        answers = []
        for output in outputs:
            # Remove the prompt from the generated text
            generated_text = tokenizer.decode(output[inputs.shape[1]:], skip_special_tokens=True)
            answers.append(generated_text.strip())
        
        results[prompt] = answers
    
    return results

def create_ranking_data(prompts_answers: Dict[str, List[str]]) -> List[Dict[str, str]]:
    """Create data structure for manual ranking"""
    ranking_data = []
    
    for prompt, answers in prompts_answers.items():
        print(f"\n{'='*60}")
        print(f"PROMPT: {prompt}")
        print(f"{'='*60}")
        
        for i, answer in enumerate(answers, 1):
            print(f"\nAnswer {i}:")
            print("-" * 40)
            print(answer)
            print("-" * 40)
        
        print("\nPlease rank these answers from 1 (best) to 4 (worst):")
        ranks = {}
        for i in range(1, 5):
            while True:
                try:
                    rank = int(input(f"Rank for Answer {i} (1-4): "))
                    if rank in [1, 2, 3, 4] and rank not in ranks.values():
                        ranks[i-1] = rank
                        break
                    else:
                        print("Please enter a unique rank between 1-4")
                except ValueError:
                    print("Please enter a valid number")
        
        # Add to ranking data
        for answer_idx, answer in enumerate(answers):
            ranking_data.append({
                'prompt': prompt,
                'answer': answer,
                'rank': ranks[answer_idx]
            })
    
    return ranking_data

def main():
    # Define 5 diverse prompts
    prompts = [
        "Tell me a funny joke about programming",
        "Write a brief summary of the benefits of renewable energy",
        "Compose a short essay about the importance of reading books",
        "Explain how to make a perfect cup of coffee",
        "Describe what makes a good friend"
    ]
    
    print("Setting up the model...")
    tokenizer, model = setup_model()
    
    print("Generating candidate answers...")
    prompts_answers = generate_answers(prompts, tokenizer, model)
    
    print("\nStarting manual ranking process...")
    ranking_data = create_ranking_data(prompts_answers)
    
    # Save to CSV
    df = pd.DataFrame(ranking_data)
    df.to_csv('training_data.csv', index=False)
    print(f"\nRanking data saved to 'training_data.csv'")
    print(f"Total samples: {len(ranking_data)}")
    
    # Save raw generated data for reference
    with open('generated_answers.json', 'w') as f:
        json.dump(prompts_answers, f, indent=2)
    
    print("Raw generated answers saved to 'generated_answers.json'")

if __name__ == "__main__":
    main() 