#!/bin/bash
#SBATCH --job-name=liullm-evaluate
#SBATCH --output=slurm_logs/evaluate_%j.out
#SBATCH --error=slurm_logs/evaluate_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --partition=gpu
#SBATCH --constraint=a100

# Print some information about the job
echo "Running on host: $(hostname)"
echo "Starting at $(date)"
echo "SLURM_JOB_ID: ${SLURM_JOB_ID}"
echo "SLURM_JOB_NODELIST: ${SLURM_JOB_NODELIST}"
echo "SLURM_CPUS_PER_TASK: ${SLURM_CPUS_PER_TASK}"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}"
nvidia-smi

# Create the log directory if it doesn't exist
mkdir -p slurm_logs

# Load necessary modules (modify these based on your cluster's configuration)
module purge
module load Anaconda3 CUDA/11.7

# Activate the conda environment
source activate liullm

# Go to the project's root directory
cd "${SLURM_SUBMIT_DIR}/.."

# Create output directories
mkdir -p outputs/evaluation
mkdir -p outputs/logs

# Parse command line arguments
CONFIG=${1:-"configs/evaluation_config.yaml"}
MODEL_PATH=${2:-"outputs/checkpoints/finetune"}
TOKENIZER_PATH=${3:-"outputs/tokenizer"}
EVAL_DATA_DIR=${4:-"data/processed/eval"}
OUTPUT_DIR=${5:-"outputs/evaluation"}
MODE=${6:-"all"}
LANGUAGE_SUBSET=${7:-""}
MAX_SAMPLES=${8:-""}
BATCH_SIZE=${9:-""}
LOG_DIR=${10:-"outputs/logs"}
JSON_OUTPUT=${11:-""}

echo "Using configuration: ${CONFIG}"
echo "Model path: ${MODEL_PATH}"
echo "Tokenizer path: ${TOKENIZER_PATH}"
echo "Evaluation data directory: ${EVAL_DATA_DIR}"
echo "Output directory: ${OUTPUT_DIR}"
echo "Evaluation mode: ${MODE}"
echo "Log directory: ${LOG_DIR}"

# Create command with base arguments
CMD="python scripts/evaluate.py \
    --config ${CONFIG} \
    --model_path ${MODEL_PATH} \
    --tokenizer_path ${TOKENIZER_PATH} \
    --eval_data_dir ${EVAL_DATA_DIR} \
    --output_dir ${OUTPUT_DIR} \
    --mode ${MODE} \
    --log_dir ${LOG_DIR}"

# Add optional parameters if provided
if [ ! -z "${LANGUAGE_SUBSET}" ]; then
    CMD="${CMD} --language_subset ${LANGUAGE_SUBSET}"
    echo "Evaluating on languages: ${LANGUAGE_SUBSET}"
fi

if [ ! -z "${MAX_SAMPLES}" ]; then
    CMD="${CMD} --max_samples ${MAX_SAMPLES}"
    echo "Using max samples: ${MAX_SAMPLES}"
fi

if [ ! -z "${BATCH_SIZE}" ]; then
    CMD="${CMD} --batch_size ${BATCH_SIZE}"
    echo "Using batch size: ${BATCH_SIZE}"
fi

# Add JSON output flag if requested
if [ "${JSON_OUTPUT}" = "true" ]; then
    CMD="${CMD} --json_output"
    echo "Using JSON output format"
fi

# Run the evaluation script
echo "Running command: ${CMD}"
eval ${CMD}

# Check if the script completed successfully
if [ $? -eq 0 ]; then
    echo "Evaluation completed successfully."
else
    echo "Evaluation failed with error code $?."
    exit 1
fi

echo "Finished at $(date)"
exit 0 