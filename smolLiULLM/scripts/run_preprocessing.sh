#!/bin/bash
# Script to run data preprocessing using the conda environment

# Activate the conda environment
eval "$(conda shell.bash hook)"
conda activate liullm

# Set base directory
REPO_ROOT="$(dirname "$(dirname "$(dirname "$0")")")"
cd "$REPO_ROOT"
echo "Working from directory: $(pwd)"

# Set variables
INPUT_DIR="$REPO_ROOT/smolLiULLM/data/raw"
OUTPUT_DIR="$REPO_ROOT/smolLiULLM/data/processed"
CONFIG_FILE="$REPO_ROOT/smolLiULLM/configs/data_config.yaml"
LOG_DIR="$REPO_ROOT/smolLiULLM/logs"
SCRIPT_PATH="$REPO_ROOT/smolLiULLM/scripts/local/run_preprocessing.py"

# Create necessary directories
mkdir -p "$OUTPUT_DIR"
mkdir -p "$LOG_DIR"

# Print status
echo "Starting data preprocessing..."
echo "Input directory: $INPUT_DIR"
echo "Output directory: $OUTPUT_DIR" 
echo "Config file: $CONFIG_FILE"
echo "Script path: $SCRIPT_PATH"

# Run the preprocessing script directly
python "$SCRIPT_PATH" \
  --config "$CONFIG_FILE" \
  --input_dir "$INPUT_DIR" \
  --output_dir "$OUTPUT_DIR" \
  --train_file "train.jsonl" \
  --val_file "val.jsonl" \
  --val_split 0.05 \
  --log_dir "$LOG_DIR"

# Print completion status
if [ $? -eq 0 ]; then
  echo "Preprocessing completed successfully!"
  echo "Processed data available in: $OUTPUT_DIR"
  echo "Language-specific directories:"
  echo "- English: $OUTPUT_DIR/english"
  echo "- Swedish: $OUTPUT_DIR/swedish"
  echo "- Code: $OUTPUT_DIR/code"
else
  echo "Error: Preprocessing failed!"
fi 