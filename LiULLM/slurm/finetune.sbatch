#!/bin/bash
#SBATCH --job-name=liullm-finetune
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=logs/finetune_%j.out
#SBATCH --error=logs/finetune_%j.err

# Create logs directory if it doesn't exist
mkdir -p logs

# Load necessary modules (modify as needed for your system)
# module load python/3.10 cuda/11.5

# Activate environment (modify as needed for your system)
source activate liullm

# Change to project directory
cd $SLURM_SUBMIT_DIR/..

# Create output directories
mkdir -p outputs/finetuned outputs/logs

# Define the checkpoint path (adjust to your actual pretrained model path)
CHECKPOINT_PATH="outputs/checkpoints/pretrain"

# Run the fine-tuning script
python scripts/run_finetuning.py \
    --config configs/finetune_config.yaml \
    --model_path $CHECKPOINT_PATH \
    --train_file data/instruction_tuning/alpaca_cleaned_train.json \
    --val_file data/instruction_tuning/alpaca_cleaned_val.json \
    --output_dir outputs/finetuned \
    --log_dir outputs/logs

echo "Fine-tuning completed!" 