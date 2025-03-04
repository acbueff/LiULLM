#!/bin/bash
#SBATCH --job-name=liullm-instruct-data
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=logs/download_instruction_data_%j.out
#SBATCH --error=logs/download_instruction_data_%j.err

# Create logs directory if it doesn't exist
mkdir -p logs

# Load necessary modules (modify as needed for your system)
# module load python/3.10

# Activate environment (modify as needed for your system)
source activate liullm

# Change to project directory
cd $SLURM_SUBMIT_DIR/..

# Create instruction tuning data directory
mkdir -p data/instruction_tuning

# Run the download script
python scripts/data_download/download_instruction_data.py \
    --dataset "yahma/alpaca-cleaned" \
    --output_dir data/instruction_tuning \
    --train_filename alpaca_cleaned_train.json \
    --val_filename alpaca_cleaned_val.json \
    --val_split 0.05 \
    --log_dir outputs/logs

echo "Instruction tuning data download completed!" 