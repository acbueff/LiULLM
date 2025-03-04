# LiULLM End-to-End Training Pipeline

This repository contains a comprehensive pipeline for training multilingual LLMs with a focus on Icelandic, Swedish, and English languages.

## Repository Structure

```
LiULLM/
├── data/                    # Raw and preprocessed datasets 
├── configs/                 # YAML or JSON config files for training hyperparameters 
├── src/                     # Source code modules
│   ├── data/                # Data loading & preprocessing scripts 
│   ├── models/              # Model architecture definition or wrappers 
│   ├── training/            # Training loop, evaluation functions
│   └── utils/               # Utility functions and helpers
├── scripts/                 # Entry-point scripts for each pipeline stage 
├── outputs/                 # Outputs like checkpoints, logs, tokenizer files 
│   ├── tokenizer/           # Saved tokenizer model/vocab 
│   ├── checkpoints/         # Model checkpoints 
│   └── logs/                # Training logs
├── slurm/                   # Slurm job scripts for HPC 
└── environment.yml          # Conda environment dependencies
```

## Getting Started

### 1. Environment Setup

```bash
# Create and activate a conda environment
conda env create -f environment.yml
conda activate liullm
```

### 2. Data Preprocessing

```bash
# Process the training data
python scripts/preprocess_data.py

# Train the tokenizer
python scripts/train_tokenizer.py
```

### 3. Model Training

For local training:
```bash
python scripts/pretrain.py
```

For HPC (Slurm) training:
```bash
sbatch slurm/pretrain.sbatch
```

### 4. Fine-tuning

```bash
python scripts/finetune.py --checkpoint outputs/checkpoints/{checkpoint_name}
```

### 5. Evaluation

```bash
python scripts/evaluate.py --model outputs/checkpoints/finetune
```

## Training Pipeline

The training pipeline consists of the following stages:

1. **Data Preprocessing**: Clean and format multilingual text data.
2. **Tokenizer Training**: Train a byte-level BPE tokenizer on the processed data.
3. **Pretraining**: Train a causal language model from scratch on the multilingual corpus.
4. **Instruction Fine-tuning**: Fine-tune the pretrained model on instruction-following data.
5. **Evaluation**: Evaluate model performance on perplexity and language-specific tasks.

## HPC Integration

This codebase is designed to work with Slurm-based HPC systems. Job scripts for data preprocessing, training, and evaluation are available in the `slurm/` directory.

## Weights & Biases Integration

The training scripts are integrated with Weights & Biases for experiment tracking. To use W&B:

```bash
# Login to W&B
wandb login

# Run training with W&B tracking
python scripts/pretrain.py --use_wandb
``` 