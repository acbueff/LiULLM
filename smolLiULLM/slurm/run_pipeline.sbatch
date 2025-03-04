#!/bin/bash
#SBATCH --job-name=liullm-pipeline
#SBATCH --output=slurm_logs/pipeline_%j.out
#SBATCH --error=slurm_logs/pipeline_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=336:00:00  # 14 days
#SBATCH --partition=cpu

# This is a master script that submits and manages the entire LiULLM training pipeline
# It will submit jobs for each stage and wait for their completion before proceeding

# Print some information about the job
echo "Running on host: $(hostname)"
echo "Starting master pipeline at $(date)"
echo "SLURM_JOB_ID: ${SLURM_JOB_ID}"

# Create the log directory if it doesn't exist
mkdir -p slurm_logs

# Load necessary modules (modify these based on your cluster's configuration)
module purge
module load Anaconda3

# Activate the conda environment
source activate liullm

# Go to the project's root directory
cd "${SLURM_SUBMIT_DIR}/.."

# Define configuration paths and output directories
DATA_CONFIG=${1:-"configs/data_config.yaml"}
TOKENIZER_CONFIG=${2:-"configs/tokenizer_config.yaml"}
PRETRAIN_CONFIG=${3:-"configs/pretrain_config.yaml"}
MODEL_CONFIG=${4:-"configs/model_config.yaml"}
FINETUNE_CONFIG=${5:-"configs/finetune_config.yaml"}
EVAL_CONFIG=${6:-"configs/evaluation_config.yaml"}
USE_WANDB=${7:-"false"}

# Create directories for outputs
mkdir -p data/processed
mkdir -p outputs/tokenizer
mkdir -p outputs/checkpoints/pretrain
mkdir -p outputs/checkpoints/finetune
mkdir -p outputs/evaluation
mkdir -p outputs/logs

echo "========== LiULLM Training Pipeline =========="
echo "Starting complete training pipeline"
echo "Data config: ${DATA_CONFIG}"
echo "Tokenizer config: ${TOKENIZER_CONFIG}"
echo "Pretrain config: ${PRETRAIN_CONFIG}"
echo "Model config: ${MODEL_CONFIG}"
echo "Fine-tune config: ${FINETUNE_CONFIG}"
echo "Evaluation config: ${EVAL_CONFIG}"
echo "Using W&B: ${USE_WANDB}"
echo "=============================================="

# Function to submit a job and wait for its completion
submit_and_wait() {
    local script=$1
    local job_name=$2
    shift 2
    
    echo "Submitting job: ${job_name}"
    echo "Command: sbatch ${script} $@"
    
    # Submit the job and capture the job ID
    local submit_output=$(sbatch ${script} "$@")
    local job_id=$(echo ${submit_output} | awk '{print $4}')
    
    echo "Submitted job ${job_name} with ID: ${job_id}"
    
    # Wait for the job to complete
    echo "Waiting for job ${job_id} to complete..."
    srun --dependency=afterany:${job_id} --job-name="wait_${job_id}" --cpus-per-task=1 --mem=100M --time=00:01:00 /bin/true
    
    # Check the job exit status
    local state=$(sacct -j ${job_id} --format=State --noheader | head -n 1 | tr -d ' ')
    if [[ "${state}" == "COMPLETED" ]]; then
        echo "Job ${job_name} (${job_id}) completed successfully"
        return 0
    else
        echo "Job ${job_name} (${job_id}) failed with state: ${state}"
        return 1
    fi
}

# 1. Data Preprocessing
echo "Step 1: Data Preprocessing"
if ! submit_and_wait slurm/preprocess_data.sbatch "preprocess" "${DATA_CONFIG}" "data/processed" "outputs/logs"; then
    echo "Data preprocessing failed. Stopping pipeline."
    exit 1
fi

# 2. Tokenizer Training
echo "Step 2: Tokenizer Training"
if ! submit_and_wait slurm/train_tokenizer.sbatch "tokenizer" "${TOKENIZER_CONFIG}" "data/processed/train.txt" "outputs/tokenizer" "32000" "outputs/logs"; then
    echo "Tokenizer training failed. Stopping pipeline."
    exit 1
fi

# 3. Pretraining
echo "Step 3: Pretraining"
pretraining_args=("${PRETRAIN_CONFIG}" "${MODEL_CONFIG}" "outputs/checkpoints/pretrain" "data/processed" "outputs/tokenizer" "" "outputs/logs")
if [ "${USE_WANDB}" = "true" ]; then
    pretraining_args+=("true")
fi

if ! submit_and_wait slurm/pretrain.sbatch "pretrain" "${pretraining_args[@]}"; then
    echo "Pretraining failed. Stopping pipeline."
    exit 1
fi

# 4. Fine-tuning
echo "Step 4: Fine-tuning"
finetuning_args=("${FINETUNE_CONFIG}" "outputs/checkpoints/pretrain-final" "outputs/tokenizer" "data/processed/instructions.jsonl" "outputs/checkpoints/finetune" "" "" "" "outputs/logs")
if [ "${USE_WANDB}" = "true" ]; then
    finetuning_args+=("true")
fi

if ! submit_and_wait slurm/finetune.sbatch "finetune" "${finetuning_args[@]}"; then
    echo "Fine-tuning failed. Stopping pipeline."
    exit 1
fi

# 5. Evaluation
echo "Step 5: Evaluation"
if ! submit_and_wait slurm/evaluate.sbatch "evaluate" "${EVAL_CONFIG}" "outputs/checkpoints/finetune" "outputs/tokenizer" "data/processed/eval" "outputs/evaluation" "all" "" "" "" "outputs/logs" "true"; then
    echo "Evaluation failed."
    # Continue even if evaluation fails
fi

echo "========== LiULLM Training Pipeline Complete =========="
echo "Pipeline completed at $(date)"
echo "Outputs available in:"
echo "- Tokenizer: outputs/tokenizer"
echo "- Pretrained model: outputs/checkpoints/pretrain-final"
echo "- Fine-tuned model: outputs/checkpoints/finetune"
echo "- Evaluation results: outputs/evaluation"
echo "- Logs: outputs/logs and slurm_logs"
echo "======================================================"

exit 0 