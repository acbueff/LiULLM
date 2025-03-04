#!/bin/bash
#SBATCH --job-name=liullm-preprocess
#SBATCH --output=slurm_logs/preprocess_%j.out
#SBATCH --error=slurm_logs/preprocess_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --partition=cpu

# Print some information about the job
echo "Running on host: $(hostname)"
echo "Starting at $(date)"
echo "SLURM_JOB_ID: ${SLURM_JOB_ID}"
echo "SLURM_JOB_NODELIST: ${SLURM_JOB_NODELIST}"
echo "SLURM_CPUS_PER_TASK: ${SLURM_CPUS_PER_TASK}"

# Create the log directory if it doesn't exist
mkdir -p slurm_logs

# Load necessary modules (modify these based on your cluster's configuration)
module purge
module load Anaconda3

# Activate the conda environment
source activate liullm

# Go to the project's root directory
cd "${SLURM_SUBMIT_DIR}/.."

# Create output directories
mkdir -p data/processed
mkdir -p outputs/logs

# Parse command line arguments
CONFIG=${1:-"configs/data_config.yaml"}
OUTPUT_DIR=${2:-"data/processed"}
LOG_DIR=${3:-"outputs/logs"}

echo "Using configuration: ${CONFIG}"
echo "Output directory: ${OUTPUT_DIR}"
echo "Log directory: ${LOG_DIR}"

# Run the data preprocessing script
python scripts/preprocess_data.py \
    --config ${CONFIG} \
    --output_dir ${OUTPUT_DIR} \
    --log_dir ${LOG_DIR} \
    --debug

# Check if the script completed successfully
if [ $? -eq 0 ]; then
    echo "Data preprocessing completed successfully."
else
    echo "Data preprocessing failed with error code $?."
    exit 1
fi

echo "Finished at $(date)"
exit 0 