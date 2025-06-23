# Reward Model Training Assignment

This project implements a complete pipeline for building and training a reward model to capture preferences in text generation, following the assignment requirements.

## Project Overview

The goal is to build a reward model that can score text responses based on quality preferences. The pipeline includes:

1. **Data Generation**: Generate 4 candidate answers for 5 prompts using a base model
2. **Ranking**: Manually rank answers from 1 (best) to 4 (worst)
3. **Training**: Train a reward model using HuggingFace TRL
4. **Evaluation**: Test the model on new data and visualize results

## Files Structure

- `requirements.txt` - Python dependencies
- `generate_data.py` - Generate candidate answers using base model (manual ranking required)
- `create_sample_data.py` - Create pre-ranked sample data for demonstration
- `reward_model_trainer.py` - Main training and evaluation script
- `training_data.csv` - Training data with prompt, answer, rank columns (generated)
- `README.md` - This file

## Installation

1. Install required packages:

```bash
pip install -r requirements.txt
```

## Usage

### Option 1: Use Pre-created Sample Data (Recommended for Quick Start)

Run the complete pipeline with sample data:

```bash
python reward_model_trainer.py
```

This will:

- Create sample training data if not present
- Train the reward model for 3 epochs
- Evaluate on test prompts
- Generate visualization plots

### Option 2: Generate Your Own Data

1. Generate candidate answers and rank them manually:

```bash
python generate_data.py
```

This will prompt you to rank answers interactively.

2. Train the reward model:

```bash
python reward_model_trainer.py
```

### Option 3: Create Sample Data Only

```bash
python create_sample_data.py
```

## Data Format

The training data CSV has the following format:

```
prompt,answer,rank
"Tell me a funny joke about programming","Why do programmers prefer dark mode? Because light attracts bugs!",1
"Tell me a funny joke about programming","Code code code.",4
...
```

- `prompt`: The input question/instruction
- `answer`: The generated response
- `rank`: Quality ranking (1=best, 4=worst)

## Model Architecture

- **Base Model**: DistilBERT (distilbert-base-uncased) - non-gated model
- **Task**: Regression (predicting reward scores)
- **Input**: "[prompt] [SEP] [answer]"
- **Output**: Single reward score

## Training Configuration

- **Epochs**: 3 (adjustable)
- **Batch Size**: 8
- **Learning Rate**: Default from Trainer
- **Train/Val Split**: 80/20
- **Evaluation**: Every 20 steps

## Evaluation

The model is evaluated on new test prompts with answers of varying quality. Results include:

1. **Reward Scores**: Numerical scores for each answer
2. **Rankings**: Answers ranked by reward scores
3. **Visualizations**:
   - Bar plots of scores by prompt/answer
   - Distribution of all scores
4. **Statistics**: Score range, mean, and standard deviation

## Expected Behavior

The trained reward model should assign:

- **Higher scores** to well-written, detailed, helpful answers
- **Lower scores** to short, vague, or poor-quality answers

## Sample Prompts Used

1. Programming jokes
2. Renewable energy summaries
3. Essays about reading importance
4. Coffee making instructions
5. Good friend qualities

## Output Files

After running the training:

- `reward_model/` - Training checkpoints
- `reward_model_final/` - Final trained model
- `logs/` - Training logs
- `reward_scores_plot.png` - Visualization
- `training_data.csv` - Training data

## Verification

The evaluation section verifies that:

1. Higher-quality answers receive higher scores
2. The model can distinguish between different quality levels
3. Scores correlate with expected preferences

## Troubleshooting

1. **Memory Issues**: Reduce batch size in `TrainingArguments`
2. **CUDA Errors**: The code works on CPU by default
3. **Module Not Found**: Ensure all requirements are installed

## Assignment Requirements Fulfilled

✅ **5 Prompts**: Diverse topics (jokes, summaries, essays, instructions, descriptions)  
✅ **4 Candidate Answers**: Generated per prompt using base model  
✅ **Ranking System**: 1-4 ranking with clear CSV format  
✅ **HuggingFace Training**: Uses Transformers library with proper trainer  
✅ **50-100 Steps**: Configurable training duration  
✅ **Evaluation**: Plots reward scores and validates correlation  
✅ **Non-gated Models**: Uses DistilBERT and DialoGPT (publicly available)
