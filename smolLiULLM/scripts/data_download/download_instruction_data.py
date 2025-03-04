#!/usr/bin/env python3
"""
Script to download instruction tuning data from Hugging Face datasets.
Specifically optimized for the yahma/alpaca-cleaned dataset.
"""

import os
import sys
import json
import argparse
import logging
from pathlib import Path
from tqdm import tqdm
from datasets import load_dataset

# Add src to path for imports
src_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "")
sys.path.insert(0, src_path)

from src.utils.logging import setup_logging

# Constants
DEFAULT_DATASET = "yahma/alpaca-cleaned"
DEFAULT_OUTPUT_DIR = "data/instruction_tuning"
DEFAULT_TRAIN_FILE = "alpaca_cleaned_train.json"
DEFAULT_VAL_FILE = "alpaca_cleaned_val.json"

logger = logging.getLogger(__name__)

def download_instruction_dataset(
    dataset_name: str,
    output_dir: str,
    train_filename: str,
    val_filename: str,
    val_split: float = 0.05,
    shuffle: bool = True,
    max_samples: int = None,
) -> tuple:
    """
    Download instruction tuning dataset from Hugging Face and save as JSON files.
    
    Args:
        dataset_name: Name of the dataset on Hugging Face
        output_dir: Directory to save the processed data
        train_filename: Filename for training data
        val_filename: Filename for validation data
        val_split: Fraction of data to use for validation
        shuffle: Whether to shuffle the data
        max_samples: Maximum number of samples to include (None for all)
    
    Returns:
        Tuple of (train_path, val_path)
    """
    logger.info(f"Downloading instruction tuning dataset: {dataset_name}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Download the dataset
    dataset = load_dataset(dataset_name)
    
    # Get the data split (usually 'train' for alpaca)
    split_name = list(dataset.keys())[0]  # Usually 'train'
    
    # Get the data
    data = dataset[split_name]
    
    # Shuffle if requested
    if shuffle:
        data = data.shuffle(seed=42)
    
    # Limit number of samples if requested
    if max_samples is not None:
        data = data.select(range(min(max_samples, len(data))))
    
    # Calculate split sizes
    val_size = int(len(data) * val_split)
    train_size = len(data) - val_size
    
    # Split the data
    train_data = data.select(range(train_size))
    val_data = data.select(range(train_size, len(data)))
    
    logger.info(f"Processing {len(train_data)} training examples and {len(val_data)} validation examples")
    
    # Convert to list of dictionaries for easier JSON serialization
    train_examples = []
    for example in tqdm(train_data, desc="Processing training data"):
        train_examples.append({
            "instruction": example["instruction"],
            "input": example["input"] if "input" in example else "",
            "output": example["output"]
        })
    
    val_examples = []
    for example in tqdm(val_data, desc="Processing validation data"):
        val_examples.append({
            "instruction": example["instruction"],
            "input": example["input"] if "input" in example else "",
            "output": example["output"]
        })
    
    # Save train data
    train_path = os.path.join(output_dir, train_filename)
    with open(train_path, "w", encoding="utf-8") as f:
        json.dump(train_examples, f, ensure_ascii=False, indent=2)
    
    # Save validation data
    val_path = os.path.join(output_dir, val_filename)
    with open(val_path, "w", encoding="utf-8") as f:
        json.dump(val_examples, f, ensure_ascii=False, indent=2)
    
    logger.info(f"Saved {len(train_examples)} training examples to {train_path}")
    logger.info(f"Saved {len(val_examples)} validation examples to {val_path}")
    
    return train_path, val_path

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Download instruction tuning data from Hugging Face")
    
    parser.add_argument(
        "--dataset",
        type=str,
        default=DEFAULT_DATASET,
        help=f"Dataset name on Hugging Face (default: {DEFAULT_DATASET})"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory to save processed data (default: {DEFAULT_OUTPUT_DIR})"
    )
    parser.add_argument(
        "--train_filename",
        type=str,
        default=DEFAULT_TRAIN_FILE,
        help=f"Filename for training data (default: {DEFAULT_TRAIN_FILE})"
    )
    parser.add_argument(
        "--val_filename",
        type=str,
        default=DEFAULT_VAL_FILE,
        help=f"Filename for validation data (default: {DEFAULT_VAL_FILE})"
    )
    parser.add_argument(
        "--val_split",
        type=float,
        default=0.05,
        help="Fraction of data to use for validation (default: 0.05)"
    )
    parser.add_argument(
        "--no_shuffle",
        action="store_true",
        help="Don't shuffle the data"
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Maximum number of samples to include (default: all)"
    )
    parser.add_argument(
        "--log_dir",
        type=str,
        default="outputs/logs",
        help="Directory to save logs"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode with more verbose logging"
    )
    
    return parser.parse_args()

def main():
    """Main function to download instruction tuning data."""
    # Parse arguments
    args = parse_args()
    
    # Set up logging
    log_level = logging.DEBUG if args.debug else logging.INFO
    logger = setup_logging(
        log_dir=args.log_dir,
        log_level=log_level,
        experiment_name="instruction_data_download"
    )
    
    try:
        # Download and process the instruction tuning data
        train_path, val_path = download_instruction_dataset(
            dataset_name=args.dataset,
            output_dir=args.output_dir,
            train_filename=args.train_filename,
            val_filename=args.val_filename,
            val_split=args.val_split,
            shuffle=not args.no_shuffle,
            max_samples=args.max_samples
        )
        
        logger.info("Instruction tuning data download completed successfully!")
        logger.info(f"Training data: {train_path}")
        logger.info(f"Validation data: {val_path}")
        
    except Exception as e:
        logger.error(f"Error downloading instruction tuning data: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main() 