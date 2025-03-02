#!/bin/bash
#SBATCH --job-name=liullm-finetune
#SBATCH --output=slurm_logs/finetune_%j.out
#SBATCH --error=slurm_logs/finetune_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=24:00:00
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
mkdir -p outputs/checkpoints/finetune
mkdir -p outputs/logs

# Parse command line arguments
CONFIG=${1:-"configs/finetune_config.yaml"}
MODEL_PATH=${2:-"outputs/checkpoints/pretrain-final"}
TOKENIZER_PATH=${3:-"outputs/tokenizer"}
DATA_PATH=${4:-"data/processed/instructions.jsonl"}
OUTPUT_DIR=${5:-"outputs/checkpoints/finetune"}
LORA_R=${6:-""}
LORA_ALPHA=${7:-""}
LORA_DROPOUT=${8:-""}
LOG_DIR=${9:-"outputs/logs"}
USE_WANDB=${10:-""}

echo "Using configuration: ${CONFIG}"
echo "Model path: ${MODEL_PATH}"
echo "Tokenizer path: ${TOKENIZER_PATH}"
echo "Data path: ${DATA_PATH}"
echo "Output directory: ${OUTPUT_DIR}"
echo "Log directory: ${LOG_DIR}"

# Create command with base arguments
CMD="python -m torch.distributed.launch --nproc_per_node=${SLURM_GPUS_PER_NODE} scripts/finetune.py \
    --config ${CONFIG} \
    --model_path ${MODEL_PATH} \
    --tokenizer_path ${TOKENIZER_PATH} \
    --data_path ${DATA_PATH} \
    --output_dir ${OUTPUT_DIR} \
    --log_dir ${LOG_DIR}"

# Add LoRA parameters if provided
if [ ! -z "${LORA_R}" ]; then
    CMD="${CMD} --lora_r ${LORA_R}"
    echo "Using LoRA r: ${LORA_R}"
fi

if [ ! -z "${LORA_ALPHA}" ]; then
    CMD="${CMD} --lora_alpha ${LORA_ALPHA}"
    echo "Using LoRA alpha: ${LORA_ALPHA}"
fi

if [ ! -z "${LORA_DROPOUT}" ]; then
    CMD="${CMD} --lora_dropout ${LORA_DROPOUT}"
    echo "Using LoRA dropout: ${LORA_DROPOUT}"
fi

# Add wandb logging if requested
if [ "${USE_WANDB}" = "true" ]; then
    CMD="${CMD} --wandb"
    echo "Using Weights & Biases logging"
fi

# Run the fine-tuning script with distributed training
echo "Running command: ${CMD}"
eval ${CMD}

# Check if the script completed successfully
if [ $? -eq 0 ]; then
    echo "Fine-tuning completed successfully."
else
    echo "Fine-tuning failed with error code $?."
    exit 1
fi

echo "Finished at $(date)"
exit 0 