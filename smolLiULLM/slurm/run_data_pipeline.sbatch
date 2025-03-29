#!/bin/bash
#SBATCH --job-name=liullm-data-pipeline
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=logs/data_pipeline_%j.out
#SBATCH --error=logs/data_pipeline_%j.err

# Create logs directory if it doesn't exist
mkdir -p logs

# Load necessary modules (modify as needed for your system)
# module load python/3.10

# Activate environment (modify as needed for your system)
source activate liullm

# Change to project directory
cd $SLURM_SUBMIT_DIR/..

# Create necessary directories
mkdir -p data/raw data/processed outputs/logs

# Run the full data pipeline
python scripts/prepare_data_pipeline.py \
    --config configs/data_config.yaml \
    --log_dir outputs/logs

echo "Data pipeline completed!" 