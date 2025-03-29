#!/bin/bash

# Script to create symbolic links from the git repository to the external data location
# Run this after cloning the repository to set up data access

# Base directories
REPO_DIR="$HOME/Documents/trustllm/TechStack/LiULLM"
DATA_DIR="$HOME/Documents/trustllm/TechStack/data/LiULLM"

# Create symbolic links for main data directories
mkdir -p "$REPO_DIR/data"
ln -sfn "$DATA_DIR/data/raw" "$REPO_DIR/data/raw"
ln -sfn "$DATA_DIR/data/processed" "$REPO_DIR/data/processed"
ln -sfn "$DATA_DIR/data/instruction_tuning" "$REPO_DIR/data/instruction_tuning"
ln -sfn "$DATA_DIR/data/tokenizer" "$REPO_DIR/data/tokenizer"

# Create symbolic links for smolLiULLM data directories
mkdir -p "$REPO_DIR/smolLiULLM/data"
ln -sfn "$DATA_DIR/smolLiULLM/data/raw" "$REPO_DIR/smolLiULLM/data/raw"
ln -sfn "$DATA_DIR/smolLiULLM/data/processed" "$REPO_DIR/smolLiULLM/data/processed"
ln -sfn "$DATA_DIR/smolLiULLM/data/instruction_tuning" "$REPO_DIR/smolLiULLM/data/instruction_tuning"
ln -sfn "$DATA_DIR/smolLiULLM/data/tokenizer" "$REPO_DIR/smolLiULLM/data/tokenizer"

# Create symbolic links for JUWELS data
mkdir -p "$REPO_DIR/JUWELS/WP2/data"
ln -sfn "$DATA_DIR/JUWELS/WP2/data/eng-common-corpus" "$REPO_DIR/JUWELS/WP2/data/eng-common-corpus"
ln -sfn "$DATA_DIR/JUWELS/WP2/data/fineweb2" "$REPO_DIR/JUWELS/WP2/data/fineweb2"
ln -sfn "$DATA_DIR/JUWELS/WP2/data/icelandic-cc" "$REPO_DIR/JUWELS/WP2/data/icelandic-cc"

# Create symbolic links for outputs
mkdir -p "$REPO_DIR"
ln -sfn "$DATA_DIR/outputs" "$REPO_DIR/outputs"
mkdir -p "$REPO_DIR/smolLiULLM"
ln -sfn "$DATA_DIR/smolLiULLM/outputs" "$REPO_DIR/smolLiULLM/outputs"

echo "Symbolic links created successfully!"
