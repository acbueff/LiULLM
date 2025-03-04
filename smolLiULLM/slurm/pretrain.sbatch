#!/bin/bash
#SBATCH --job-name=liullm-pretrain
#SBATCH --output=slurm_logs/pretrain_%j.out
#SBATCH --error=slurm_logs/pretrain_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=72:00:00
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
mkdir -p outputs/checkpoints
mkdir -p outputs/logs

# Parse command line arguments
CONFIG=${1:-"configs/pretrain_config.yaml"}
MODEL_CONFIG=${2:-"configs/model_config.yaml"}
OUTPUT_DIR=${3:-"outputs/checkpoints/pretrain"}
DATA_DIR=${4:-"data/processed"}
TOKENIZER_PATH=${5:-"outputs/tokenizer"}
RESUME_FROM=${6:-""}
LOG_DIR=${7:-"outputs/logs"}
USE_WANDB=${8:-""}

echo "Using training configuration: ${CONFIG}"
echo "Using model configuration: ${MODEL_CONFIG}"
echo "Output directory: ${OUTPUT_DIR}"
echo "Data directory: ${DATA_DIR}"
echo "Tokenizer path: ${TOKENIZER_PATH}"
echo "Log directory: ${LOG_DIR}"

# Create command with base arguments
CMD="python -m torch.distributed.launch --nproc_per_node=${SLURM_GPUS_PER_NODE} scripts/pretrain.py \
    --config ${CONFIG} \
    --model_config ${MODEL_CONFIG} \
    --output_dir ${OUTPUT_DIR} \
    --data_dir ${DATA_DIR} \
    --tokenizer_path ${TOKENIZER_PATH} \
    --log_dir ${LOG_DIR}"

# Add resume from checkpoint if provided
if [ ! -z "${RESUME_FROM}" ]; then
    CMD="${CMD} --resume_from_checkpoint ${RESUME_FROM}"
    echo "Resuming from checkpoint: ${RESUME_FROM}"
fi

# Add wandb logging if requested
if [ "${USE_WANDB}" = "true" ]; then
    CMD="${CMD} --wandb"
    echo "Using Weights & Biases logging"
fi

# Run the pretraining script with distributed training
echo "Running command: ${CMD}"
eval ${CMD}

# Check if the script completed successfully
if [ $? -eq 0 ]; then
    echo "Pretraining completed successfully."
else
    echo "Pretraining failed with error code $?."
    exit 1
fi

echo "Finished at $(date)"
exit 0 