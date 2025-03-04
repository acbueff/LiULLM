#!/bin/bash
#SBATCH --job-name=liullm-tokenizer
#SBATCH --output=slurm_logs/tokenizer_%j.out
#SBATCH --error=slurm_logs/tokenizer_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=12:00:00
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
mkdir -p outputs/tokenizer
mkdir -p outputs/logs

# Parse command line arguments
CONFIG=${1:-"configs/tokenizer_config.yaml"}
INPUT_FILE=${2:-"data/processed/train.txt"}
OUTPUT_DIR=${3:-"outputs/tokenizer"}
VOCAB_SIZE=${4:-32000}
LOG_DIR=${5:-"outputs/logs"}

echo "Using configuration: ${CONFIG}"
echo "Input file: ${INPUT_FILE}"
echo "Output directory: ${OUTPUT_DIR}"
echo "Vocabulary size: ${VOCAB_SIZE}"
echo "Log directory: ${LOG_DIR}"

# Run the tokenizer training script
python scripts/train_tokenizer.py \
    --config ${CONFIG} \
    --input_file ${INPUT_FILE} \
    --output_dir ${OUTPUT_DIR} \
    --vocab_size ${VOCAB_SIZE} \
    --log_dir ${LOG_DIR} \
    --debug

# Check if the script completed successfully
if [ $? -eq 0 ]; then
    echo "Tokenizer training completed successfully."
else
    echo "Tokenizer training failed with error code $?."
    exit 1
fi

echo "Finished at $(date)"
exit 0 