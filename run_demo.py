#!/usr/bin/env python3
"""
Demo script for the Reward Model Training Assignment
This script runs the complete pipeline with minimal setup required.
"""

import os
import sys

def main():
    print("🚀 Reward Model Training Demo")
    print("=" * 50)
    
    print("\n📋 Step 1: Creating sample training data...")
    try:
        import create_sample_data
        create_sample_data.main()
        print("✅ Sample data created successfully!")
    except Exception as e:
        print(f"❌ Error creating sample data: {e}")
        return
    
    print("\n🧠 Step 2: Training the reward model...")
    try:
        import reward_model_trainer
        reward_model_trainer.main()
        print("✅ Model training completed successfully!")
    except Exception as e:
        print(f"❌ Error during training: {e}")
        return
    
    print("\n🎉 Demo completed successfully!")
    print("\nFiles created:")
    files_to_check = [
        "training_data.csv",
        "reward_scores_plot.png",
        "reward_model_final"
    ]
    
    for file in files_to_check:
        if os.path.exists(file):
            print(f"✅ {file}")
        else:
            print(f"❌ {file} (not found)")
    
    print("\n📊 Check 'reward_scores_plot.png' for visualization results!")
    print("📁 The trained model is saved in 'reward_model_final/' directory")

if __name__ == "__main__":
    main() 