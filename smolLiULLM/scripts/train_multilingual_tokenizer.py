#!/usr/bin/env python3
"""
Script for training a multilingual tokenizer using data from multiple languages
and domains (English, Swedish, and code) for use with the LiULLM model.
"""

import os
import sys
import argparse
import logging
import json
from typing import Dict, Any

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ""))

from src.utils.logging import setup_logging
from src.utils.config import load_config
from src.data.multilingual_tokenization import MultilingualTokenizerTrainer, parse_sampling_ratios

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train a multilingual tokenizer")
    
    parser.add_argument(
        "--config", 
        type=str, 
        default="configs/tokenizer_config.yaml",
        help="Path to tokenizer configuration file"
    )
    parser.add_argument(
        "--data_root", 
        type=str, 
        default="data/processed",
        help="Root directory with training data subdirectories"
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default="data/tokenizer",
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
        "--samples_per_type", 
        type=str, 
        default="english=0.4,swedish=0.4,code=0.2",
        help="Sampling ratio for each dataset type, in format 'type=weight,type=weight'"
    )
    parser.add_argument(
        "--model_type",
        type=str,
        default="bpe",
        choices=["bpe", "unigram", "wordpiece", "char"],
        help="Tokenizer model type"
    )
    parser.add_argument(
        "--byte_level",
        action="store_true",
        help="Use byte-level BPE tokenization"
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
    """Main function to train a multilingual tokenizer."""
    # Parse arguments
    args = parse_args()
    
    # Set up logging
    log_level = logging.DEBUG if args.debug else logging.INFO
    logger = setup_logging(
        log_dir=args.log_dir,
        log_level=log_level,
        experiment_name="multilingual_tokenizer_training"
    )
    
    try:
        # Load configuration if provided
        if os.path.exists(args.config):
            logger.info(f"Loading configuration from {args.config}")
            config = load_config(args.config)
        else:
            logger.info("No config file found, using command line arguments")
            config = {}
        
        # Parse sampling ratios
        sampling_ratios = parse_sampling_ratios(args.samples_per_type)
        logger.info(f"Using sampling ratios: {sampling_ratios}")
        
        # Setup tokenizer training configuration
        tokenizer_config = {
            'model_type': args.model_type,
            'byte_level': args.byte_level,
            'vocab_size': args.vocab_size,
            'data_root': args.data_root,
            'output_dir': args.output_dir,
            'max_samples_per_dataset': args.max_samples_per_dataset,
            'sampling_ratios': sampling_ratios,
            'special_tokens': ["<s>", "</s>", "<unk>", "<pad>", "<mask>"]
        }
        
        # Update with config file values if not overridden by command line
        for key in config:
            if key not in tokenizer_config:
                tokenizer_config[key] = config[key]
        
        # Create output directory
        os.makedirs(args.output_dir, exist_ok=True)
        
        # Initialize multilingual tokenizer trainer
        logger.info("Initializing multilingual tokenizer trainer...")
        trainer = MultilingualTokenizerTrainer(tokenizer_config)
        
        # Train tokenizer
        logger.info("Starting tokenizer training...")
        tokenizer = trainer.train_from_multilingual()
        
        # Save tokenizer
        logger.info("Saving tokenizer...")
        trainer.save_tokenizer(tokenizer)
        
        # Save configuration for reproducibility
        config_file = os.path.join(args.output_dir, "multilingual_tokenizer_config.json")
        with open(config_file, 'w') as f:
            json.dump(tokenizer_config, f, indent=2)
        
        logger.info(f"Tokenizer configuration saved to {config_file}")
        logger.info("Multilingual tokenizer training completed successfully!")
        
    except Exception as e:
        logger.error(f"Error in tokenizer training: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main() 