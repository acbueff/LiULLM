#!/bin/bash
#SBATCH --job-name=liullm-data-download
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=logs/download_data_%j.out
#SBATCH --error=logs/download_data_%j.err

# Create logs directory if it doesn't exist
mkdir -p logs

# Load necessary modules (modify as needed for your system)
# module load python/3.10

# Activate environment (modify as needed for your system)
source activate liullm

# Change to project directory
cd $SLURM_SUBMIT_DIR/..

# Create directories
mkdir -p data/raw/eng data/raw/swe data/raw/isl

# Run the download script
python scripts/data_download/download_datasets.py \
    --output_dir data/raw \
    --languages all \
    --english_size_gb 5 \
    --swedish_min_gb 2.5 \
    --swedish_max_gb 3 \
    --icelandic_size_gb 2 \
    --log_dir outputs/logs

echo "Data download completed!" 