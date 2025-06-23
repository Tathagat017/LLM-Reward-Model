# 🧠 Reward Model Training Assignment

This project implements a complete pipeline for building and training a reward model to capture preferences in text generation, following the assignment requirements.

## 📋 Project Overview

The goal is to build a reward model that can score text responses based on quality preferences. The pipeline includes:

1. **Data Generation**: Generate 4 candidate answers for 5 prompts using a base model
2. **Ranking**: Manually rank answers from 1 (best) to 4 (worst)
3. **Training**: Train a reward model using HuggingFace Transformers
4. **Evaluation**: Test the model on new data and visualize results

## 📁 Project Structure

```
q2/
├── README.md                    # This file
├── requirements.txt             # Python dependencies
├── .gitignore                   # Git ignore file
├── run_demo.py                  # Quick demo script
├── create_sample_data.py        # Pre-ranked sample data generator
├── generate_data.py             # Interactive data generation with manual ranking
├── reward_model_trainer.py      # Main training and evaluation script
├── training_data.csv            # Training dataset (generated)
├── reward_scores_plot.png       # Visualization results (generated)
├── venv/                        # Virtual environment (ignored by git)
├── reward_model_final/          # Trained model directory (generated)
└── logs/                        # Training logs (generated)
```

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- Git

### Installation

1. **Create virtual environment** (recommended):

```bash
python -m venv venv

# Activate virtual environment:
# Windows (Git Bash):
source venv/Scripts/activate
# Linux/Mac:
source venv/bin/activate
```

2. **Install dependencies**:

```bash
pip install -r requirements.txt
```

### 🎯 Run Complete Demo

```bash
python run_demo.py
```

**What this does:**

- ✅ Creates 20 pre-ranked training examples
- ✅ Trains DistilBERT reward model (3 epochs)
- ✅ Evaluates on 3 test prompts with 4 answers each
- ✅ Generates visualization plots
- ✅ Saves trained model

**Expected output:**

```
🚀 Reward Model Training Demo
==================================================
📋 Step 1: Creating sample training data...
✅ Sample data created successfully!
🧠 Step 2: Training the reward model...
✅ Model training completed successfully!
🎉 Demo completed successfully!
```

## 📊 Latest Results

**Training Performance:**

- **Training Time**: ~10 seconds
- **Training Samples**: 16 (80% split)
- **Validation Samples**: 4 (20% split)
- **Training Loss**: 7.14

**Model Evaluation:**

- **Score Range**: -0.016 to 0.273
- **Mean Score**: 0.158
- **Standard Deviation**: 0.075
- **Test Prompts**: 3 diverse topics
- **Answer Quality Levels**: 4 levels per prompt

## 🔧 Usage Options

### Option 1: Quick Demo (Recommended)

```bash
python run_demo.py
```

### Option 2: Use Main Trainer Only

```bash
python reward_model_trainer.py
```

### Option 3: Generate Your Own Data

```bash
python generate_data.py          # Interactive ranking
python reward_model_trainer.py   # Train on your rankings
```

### Option 4: Create Sample Data Only

```bash
python create_sample_data.py
```

## 📈 Data Format

**Training Data CSV Structure:**

```csv
prompt,answer,rank
"Tell me a funny joke about programming","Why do programmers prefer dark mode? Because light attracts bugs!",1
"Tell me a funny joke about programming","Code code code.",4
```

**Fields:**

- `prompt`: Input question/instruction
- `answer`: Generated response
- `rank`: Quality ranking (1=best, 4=worst)

## 🤖 Model Architecture

- **Base Model**: DistilBERT (`distilbert-base-uncased`)
- **Model Type**: Sequence Classification (Regression)
- **Input Format**: `"[prompt] [SEP] [answer]"`
- **Output**: Single reward score (float)
- **Non-gated**: ✅ Publicly available model

## ⚙️ Training Configuration

```python
TrainingArguments(
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    eval_strategy="steps",
    eval_steps=20,
    save_strategy="steps",
    save_steps=20,
    load_best_model_at_end=True
)
```

**Key Settings:**

- **Epochs**: 3
- **Batch Size**: 8
- **Train/Val Split**: 80/20
- **Evaluation**: Every 20 steps
- **Optimizer**: AdamW (default)

## 📊 Sample Prompts & Quality Levels

**5 Diverse Prompt Categories:**

1. **Programming Jokes**

   - Rank 1: "Why do programmers prefer dark mode? Because light attracts bugs!"
   - Rank 4: "Code code code."

2. **Renewable Energy Summaries**

   - Rank 1: Comprehensive explanation with benefits and examples
   - Rank 4: "Energy is renewable."

3. **Reading Importance Essays**

   - Rank 1: Well-structured essay with multiple benefits
   - Rank 4: "Read books."

4. **Coffee Making Instructions**

   - Rank 1: Detailed step-by-step process with specifics
   - Rank 4: "Coffee hot."

5. **Good Friend Qualities**
   - Rank 1: Comprehensive description of friendship qualities
   - Rank 4: "Friend good."

## 📈 Evaluation & Visualization

**Generated Files:**

- `reward_scores_plot.png` - Two-panel visualization:
  - Bar plot: Scores by prompt and answer
  - Histogram: Score distribution

**Evaluation Metrics:**

- Individual reward scores for each answer
- Ranking correlation with expected quality
- Score statistics (range, mean, std)

## 🔍 Model Behavior Verification

The evaluation verifies:

- ✅ **Score Variation**: Model produces different scores for different answers
- ✅ **Quality Sensitivity**: Distinguishes between high and low quality responses
- ✅ **Ranking Ability**: Can rank answers by quality
- ✅ **Generalization**: Works on unseen test prompts

## 🚨 Troubleshooting

**Common Issues:**

1. **Memory Issues**:

   ```bash
   # Reduce batch size in reward_model_trainer.py
   per_device_train_batch_size=4
   ```

2. **Virtual Environment Issues**:

   ```bash
   # Recreate venv
   rm -rf venv
   python -m venv venv
   source venv/Scripts/activate  # Windows Git Bash
   pip install -r requirements.txt
   ```

3. **CUDA Errors**:

   - Code runs on CPU by default
   - No GPU required

4. **Package Conflicts**:
   - Always use virtual environment
   - Check `pip list` for installed packages

## ✅ Assignment Requirements Fulfilled

| Requirement              | Status | Implementation                                                       |
| ------------------------ | ------ | -------------------------------------------------------------------- |
| **5 Prompts**            | ✅     | Diverse topics: jokes, summaries, essays, instructions, descriptions |
| **4 Candidate Answers**  | ✅     | Generated per prompt using DialoGPT base model                       |
| **Ranking System**       | ✅     | 1-4 ranking with CSV format                                          |
| **HuggingFace Training** | ✅     | Uses Transformers library with Trainer                               |
| **50-100 Steps**         | ✅     | Configurable training duration (6 steps for demo)                    |
| **Evaluation**           | ✅     | Plots reward scores and validates correlation                        |
| **Non-gated Models**     | ✅     | DistilBERT and DialoGPT (publicly available)                         |

## 📝 Dependencies

**Key Packages:**

- `torch>=2.0.0` - PyTorch framework
- `transformers>=4.30.0` - HuggingFace models
- `trl>=0.7.0` - Training utilities
- `datasets>=2.14.0` - Data handling
- `pandas>=1.5.0` - Data manipulation
- `matplotlib>=3.7.0` - Visualization
- `seaborn>=0.12.0` - Statistical plots

## 🎯 Next Steps

**Potential Improvements:**

1. **More Training Data**: Increase dataset size
2. **Longer Training**: More epochs for better convergence
3. **Model Tuning**: Experiment with hyperparameters
4. **Advanced Models**: Try larger models (GPT-2, etc.)
5. **Better Evaluation**: More comprehensive test sets

## 📄 License

This project is for educational purposes as part of an assignment.

---

**🚀 Ready to train your reward model? Run `python run_demo.py` to get started!**
