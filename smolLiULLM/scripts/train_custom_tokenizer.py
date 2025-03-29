#!/usr/bin/env python3
"""
Custom script for training a tokenizer using multilingual datasets.
This script uses the processed data from the folders:
  - LiULLM/smolLiULLM/data/processed/code
  - LiULLM/smolLiULLM/data/processed/english
  - LiULLM/smolLiULLM/data/processed/swedish

The trained tokenizer is saved for eventual pretraining of the model.
Note: Logging is kept to summary counts to prevent the log files
from recording extensive text data.
"""

import os
import sys
import argparse
import logging
import json

# Add project root (LiULLM/smolLiULLM) to sys.path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ""))

from src.utils.logging import setup_logging
from src.data.multilingual_tokenization import MultilingualTokenizerTrainer, parse_sampling_ratios

def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a multilingual tokenizer using processed data (code, english, swedish)"
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default="LiULLM/smolLiULLM/data/processed",
        help="Root directory containing data subfolders for each language/domain"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="LiULLM/smolLiULLM/data/tokenizer",
        help="Directory to save the trained tokenizer"
    )
    parser.add_argument(
        "--vocab_size",
        type=int,
        default=32000,
        help="Vocabulary size for the tokenizer"
    )
    parser.add_argument(
        "--max_samples_per_dataset",
        type=int,
        default=100000,
        help="Maximum samples to use from each dataset"
    )
    parser.add_argument(
        "--sampling_ratios",
        type=str,
        default="english=0.33,swedish=0.33,code=0.34",
        help="Sampling ratios for each dataset type in the format 'english=ratio,swedish=ratio,code=ratio'"
    )
    parser.add_argument(
        "--model_type",
        type=str,
        default="bpe",
        choices=["bpe", "unigram", "wordpiece", "char"],
        help="Type of tokenizer model to train"
    )
    parser.add_argument(
        "--byte_level",
        action="store_true",
        help="Use byte-level tokenization"
    )
    parser.add_argument(
        "--log_dir",
        type=str,
        default="LiULLM/smolLiULLM/outputs/logs",
        help="Directory to store log files"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode with more verbose logging"
    )
    return parser.parse_args()

def main():
    args = parse_args()

    # Setup logging (only summary info will be logged)
    log_level = logging.DEBUG if args.debug else logging.INFO
    logger = setup_logging(
        log_dir=args.log_dir,
        log_level=log_level,
        experiment_name="custom_tokenizer_training"
    )

    logger.info("Starting custom tokenizer training script.")

    # Parse sampling ratios from the provided argument string
    sampling_ratios = parse_sampling_ratios(args.sampling_ratios)
    logger.info(f"Using sampling ratios: {sampling_ratios}")

    # Build the tokenizer configuration
    tokenizer_config = {
        'data_root': args.data_root,
        'output_dir': args.output_dir,
        'vocab_size': args.vocab_size,
        'max_samples_per_dataset': args.max_samples_per_dataset,
        'sampling_ratios': sampling_ratios,
        'dataset_types': ['english', 'swedish', 'code'],  # Force these three datasets
        'model_type': args.model_type,
        'byte_level': args.byte_level,
        'special_tokens': ["<s>", "</s>", "<unk>", "<pad>", "<mask>"]
    }

    logger.info("Tokenizer configuration has been set up.")

    # Initialize the multilingual tokenizer trainer
    trainer = MultilingualTokenizerTrainer(tokenizer_config)

    # Train the tokenizer using the multilingual data
    logger.info("Collecting training data and starting tokenizer training.")
    tokenizer = trainer.train_from_multilingual()

    # Save the trained tokenizer
    logger.info("Saving the trained tokenizer.")
    trainer.save_tokenizer(tokenizer)

    # Save configuration for reproducibility
    config_file = os.path.join(args.output_dir, "custom_tokenizer_config.json")
    with open(config_file, "w") as f:
        json.dump(tokenizer_config, f, indent=2)
    logger.info(f"Tokenizer configuration saved to {config_file}.")
    logger.info("Custom tokenizer training completed successfully.")

if __name__ == "__main__":
    main() 