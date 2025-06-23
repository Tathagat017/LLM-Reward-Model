import torch
import pandas as pd
import numpy as np
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification, 
    TrainingArguments,
    Trainer
)
from datasets import Dataset
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple
import os

class RewardModelTrainer:
    def __init__(self, model_name: str = "distilbert-base-uncased"):
        """Initialize the reward model trainer"""
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self.trained_model = None
        
    def setup_model(self):
        """Setup tokenizer and model for reward modeling"""
        print(f"Setting up model: {self.model_name}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=1  # Regression task for reward scores
        )
        
        # Add padding token if it doesn't exist
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
    def prepare_data(self, csv_file: str) -> Tuple[Dataset, Dataset]:
        """Prepare training data from CSV file"""
        print("Loading and preparing training data...")
        
        # Load data
        df = pd.read_csv(csv_file)
        
        # Convert ranks to reward scores (higher rank = lower reward)
        # Rank 1 (best) -> Reward 4, Rank 4 (worst) -> Reward 1
        df['reward_score'] = 5 - df['rank']  # Maps 1->4, 2->3, 3->2, 4->1
        
        # Combine prompt and answer
        df['text'] = df['prompt'] + " [SEP] " + df['answer']
        
        # Split data
        train_texts, val_texts, train_scores, val_scores = train_test_split(
            df['text'].tolist(),
            df['reward_score'].tolist(),
            test_size=0.2,
            random_state=42
        )
        
        # Tokenize data
        train_encodings = self.tokenizer(
            train_texts,
            truncation=True,
            padding=True,
            max_length=512,
            return_tensors='pt'
        )
        
        val_encodings = self.tokenizer(
            val_texts,
            truncation=True,
            padding=True,
            max_length=512,
            return_tensors='pt'
        )
        
        # Create datasets
        train_dataset = Dataset.from_dict({
            'input_ids': train_encodings['input_ids'],
            'attention_mask': train_encodings['attention_mask'],
            'labels': torch.tensor(train_scores, dtype=torch.float)
        })
        
        val_dataset = Dataset.from_dict({
            'input_ids': val_encodings['input_ids'],
            'attention_mask': val_encodings['attention_mask'],
            'labels': torch.tensor(val_scores, dtype=torch.float)
        })
        
        print(f"Training samples: {len(train_dataset)}")
        print(f"Validation samples: {len(val_dataset)}")
        
        return train_dataset, val_dataset
    
    def train_model(self, train_dataset: Dataset, val_dataset: Dataset, num_train_epochs: int = 3):
        """Train the reward model"""
        print("Starting model training...")
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir='./reward_model',
            num_train_epochs=num_train_epochs,
            per_device_train_batch_size=8,
            per_device_eval_batch_size=8,
            warmup_steps=10,
            weight_decay=0.01,
            logging_dir='./logs',
            logging_steps=10,
            eval_strategy="steps",
            eval_steps=20,
            save_strategy="steps",  # Changed to match eval_strategy
            save_steps=20,          # Added save_steps to match eval_steps
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
        )
        
        # Custom trainer for regression
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            tokenizer=self.tokenizer,
        )
        
        # Train the model
        trainer.train()
        
        # Save the trained model
        self.trained_model = trainer.model
        trainer.save_model('./reward_model_final')
        
        print("Training completed!")
        return trainer
    
    def evaluate_model(self, test_prompts: List[str], test_answers: List[List[str]]) -> Dict:
        """Evaluate the trained model on new data"""
        print("Evaluating model...")
        
        if self.trained_model is None:
            print("No trained model found. Loading from saved model...")
            self.trained_model = AutoModelForSequenceClassification.from_pretrained('./reward_model_final')
        
        results = {}
        all_scores = []
        
        for prompt, answers in zip(test_prompts, test_answers):
            scores = []
            
            for answer in answers:
                # Prepare input
                text = prompt + " [SEP] " + answer
                inputs = self.tokenizer(
                    text,
                    return_tensors='pt',
                    truncation=True,
                    padding=True,
                    max_length=512
                )
                
                # Get prediction
                with torch.no_grad():
                    outputs = self.trained_model(**inputs)
                    score = outputs.logits.item()
                    scores.append(score)
            
            results[prompt] = {
                'answers': answers,
                'scores': scores
            }
            all_scores.extend(scores)
        
        return results, all_scores
    
    def plot_results(self, results: Dict, save_path: str = 'reward_scores_plot.png'):
        """Plot reward scores for visualization"""
        print("Creating visualization...")
        
        # Prepare data for plotting
        plot_data = []
        for prompt_idx, (prompt, data) in enumerate(results.items()):
            for answer_idx, (answer, score) in enumerate(zip(data['answers'], data['scores'])):
                plot_data.append({
                    'Prompt': f"Prompt {prompt_idx + 1}",
                    'Answer': f"Answer {answer_idx + 1}",
                    'Score': score,
                    'Answer_Text': answer[:50] + "..." if len(answer) > 50 else answer
                })
        
        df_plot = pd.DataFrame(plot_data)
        
        # Create figure with subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot 1: Scores by prompt and answer
        sns.barplot(data=df_plot, x='Answer', y='Score', hue='Prompt', ax=ax1)
        ax1.set_title('Reward Scores by Prompt and Answer')
        ax1.set_xlabel('Answer Number')
        ax1.set_ylabel('Reward Score')
        ax1.legend(title='Prompt', bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # Plot 2: Distribution of all scores
        ax2.hist(df_plot['Score'], bins=20, alpha=0.7, edgecolor='black')
        ax2.set_title('Distribution of Reward Scores')
        ax2.set_xlabel('Reward Score')
        ax2.set_ylabel('Frequency')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        # Print detailed results
        print("\nDetailed Results:")
        print("=" * 80)
        for prompt, data in results.items():
            print(f"\nPrompt: {prompt}")
            print("-" * 60)
            sorted_pairs = sorted(zip(data['answers'], data['scores']), 
                                key=lambda x: x[1], reverse=True)
            for i, (answer, score) in enumerate(sorted_pairs, 1):
                print(f"Rank {i} (Score: {score:.3f}): {answer[:100]}...")
        
        return df_plot

def main():
    # Initialize trainer
    trainer = RewardModelTrainer()
    trainer.setup_model()
    
    # Check if training data exists
    if not os.path.exists('training_data.csv'):
        print("Training data not found. Creating sample data...")
        import create_sample_data
        create_sample_data.main()
    
    # Prepare data
    train_dataset, val_dataset = trainer.prepare_data('training_data.csv')
    
    # Train model
    trained_trainer = trainer.train_model(train_dataset, val_dataset, num_train_epochs=3)
    
    # Prepare test data for evaluation
    test_prompts = [
        "Write a creative story about space exploration",
        "Explain the concept of artificial intelligence",
        "Describe your favorite vacation destination"
    ]
    
    test_answers = [
        [
            "Captain Sarah gazed out at the swirling nebula, her heart racing with anticipation. After years of preparation, humanity's first interstellar mission was about to make contact with an alien civilization. The ship's AI announced: 'Unknown vessel detected. Initiating first contact protocol.' This was the moment that would change everything.",
            "Space is big and has stars. People want to go there with rockets.",
            "Astronauts go to space in spaceships to explore planets and find aliens maybe.",
            "In the year 2157, the starship Endeavor approached the mysterious planet Kepler-442b. Advanced propulsion systems and AI navigation made this 1,200-light-year journey possible, marking humanity's greatest achievement."
        ],
        [
            "Artificial Intelligence (AI) refers to the simulation of human intelligence in machines designed to think and act like humans. It encompasses machine learning, natural language processing, computer vision, and robotics. AI systems can analyze data, recognize patterns, make decisions, and continuously improve their performance through experience.",
            "AI is when computers are smart like people and can do things by themselves.",
            "Artificial intelligence means making machines think. It uses algorithms and data to solve problems.",
            "AI computer smart."
        ],
        [
            "My favorite destination is the Amalfi Coast in Italy, where dramatic cliffs meet azure Mediterranean waters. The picturesque towns of Positano and Ravello offer stunning architecture, world-class cuisine, and breathtaking sunset views. The combination of rich history, natural beauty, and Italian hospitality creates an unforgettable experience.",
            "I like going to the beach because it's fun and relaxing.",
            "Nice place with good food and pretty views.",
            "Vacation good."
        ]
    ]
    
    # Evaluate model
    results, all_scores = trainer.evaluate_model(test_prompts, test_answers)
    
    # Plot results
    plot_df = trainer.plot_results(results)
    
    print(f"\nModel evaluation completed!")
    print(f"Score range: {min(all_scores):.3f} to {max(all_scores):.3f}")
    print(f"Mean score: {np.mean(all_scores):.3f}")
    print(f"Standard deviation: {np.std(all_scores):.3f}")

if __name__ == "__main__":
    main() 