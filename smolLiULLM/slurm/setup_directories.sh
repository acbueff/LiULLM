#!/bin/bash
# Helper script to set up all necessary directories for the LiULLM pipeline

echo "Setting up directories for LiULLM pipeline..."

# Create base directories
mkdir -p data/raw/isl
mkdir -p data/raw/swe
mkdir -p data/raw/eng
mkdir -p data/processed
mkdir -p data/processed/eval

# Create output directories
mkdir -p outputs/tokenizer
mkdir -p outputs/checkpoints/pretrain
mkdir -p outputs/checkpoints/finetune
mkdir -p outputs/evaluation
mkdir -p outputs/logs

# Create Slurm log directories
mkdir -p slurm_logs

echo "Directory structure created successfully!"
echo ""
echo "Next steps:"
echo "1. Add raw data to data/raw/{isl,swe,eng} directories"
echo "2. Run the pipeline using:"
echo "   sbatch slurm/run_pipeline.sbatch"
echo ""
echo "Or run individual stages:"
echo "- sbatch slurm/preprocess_data.sbatch"
echo "- sbatch slurm/train_tokenizer.sbatch"
echo "- sbatch slurm/pretrain.sbatch"
echo "- sbatch slurm/finetune.sbatch"
echo "- sbatch slurm/evaluate.sbatch"

# Make sure all the Slurm scripts are executable
chmod +x slurm/*.sbatch

echo ""
echo "Made all Slurm scripts executable!" 