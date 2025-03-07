#!/usr/bin/env python3
"""
Script for training a balanced multilingual tokenizer using data from
English, Swedish, and code sources for the LiULLM model.

This script uses the MultilingualTokenizerTrainer to create a tokenizer
with balanced representation of multiple languages/domains.
"""

import os
import sys
import argparse
import logging
from pathlib import Path

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.logging import setup_logging
from src.utils.config import load_config
from src.data.multilingual_tokenization import MultilingualTokenizerTrainer, parse_sampling_ratios

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train a balanced multilingual tokenizer")
    
    parser.add_argument(
        "--config", 
        type=str, 
        default="configs/tokenizer_config.yaml",
        help="Path to tokenizer configuration file"
    )
    parser.add_argument(
        "--data_root", 
        type=str, 
        default="smolLiULLM/data/processed",
        help="Root directory containing language subdirectories (english, swedish, code)"
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default="smolLiULLM/data/tokenizer",
        help="Directory to save the trained tokenizer"
    )
    parser.add_argument(
        "--vocab_size", 
        type=int, 
        default=32000,
        help="Vocabulary size for the tokenizer"
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
        "--max_samples", 
        type=int, 
        default=100000,
        help="Maximum samples to use from each dataset type"
    )
    parser.add_argument(
        "--sampling_ratios", 
        type=str, 
        default="english=0.4,swedish=0.4,code=0.2",
        help="Sampling ratio for each dataset type (format: 'type=weight,type=weight')"
    )
    parser.add_argument(
        "--log_dir", 
        type=str, 
        default="smolLiULLM/outputs/logs",
        help="Directory to save logs"
    )
    parser.add_argument(
        "--log_level", 
        type=str, 
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging level"
    )
    
    return parser.parse_args()

def setup_filtered_logging(log_dir, log_level):
    """
    Set up logging with filtering to prevent data text from being logged.
    
    Args:
        log_dir: Directory to save logs
        log_level: Logging level
    
    Returns:
        Logger instance
    """
    os.makedirs(log_dir, exist_ok=True)
    
    # Convert string log level to numeric value
    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f"Invalid log level: {log_level}")
    
    # Set up root logger
    logger = logging.getLogger()
    logger.setLevel(numeric_level)
    
    # Clear any existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Console handler
    console = logging.StreamHandler()
    console.setLevel(numeric_level)
    console_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console.setFormatter(console_formatter)
    logger.addHandler(console)
    
    # File handler
    log_file = os.path.join(log_dir, "tokenizer_training.log")
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(numeric_level)
    file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)
    
    # Add filter to prevent logging of text data
    class TextFilter(logging.Filter):
        def filter(self, record):
            # Filter out potentially large text data in log messages
            msg = str(record.msg)
            if len(msg) > 1000 and ('\n' in msg or len(msg.split()) > 100):
                record.msg = f"{msg[:500]}... [truncated, total length: {len(msg)}]"
            return True
    
    text_filter = TextFilter()
    console.addFilter(text_filter)
    file_handler.addFilter(text_filter)
    
    return logger

def find_data_directory(base_path):
    """
    Find the correct data directory by checking various possible paths.
    
    Args:
        base_path: The base path to check
        
    Returns:
        The correct data directory path
    """
    # Check if the base path exists
    if os.path.exists(base_path):
        return base_path
    
    # Try with smolLiULLM prefix if not already there
    if not base_path.startswith("smolLiULLM"):
        smolliullm_path = os.path.join("smolLiULLM", base_path)
        if os.path.exists(smolliullm_path):
            return smolliullm_path
    
    # Try absolute path from project root
    script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    project_root = os.path.dirname(script_dir)
    
    # Try with data/processed
    absolute_path = os.path.join(project_root, "data", "processed")
    if os.path.exists(absolute_path):
        return absolute_path
    
    # Try with smolLiULLM/data/processed
    absolute_smol_path = os.path.join(project_root, "smolLiULLM", "data", "processed")
    if os.path.exists(absolute_smol_path):
        return absolute_smol_path
    
    # Return the original path if no alternatives are found
    return base_path

def main():
    """Main function to train a balanced multilingual tokenizer."""
    # Parse arguments
    args = parse_args()
    
    # Set up logging with filtering to prevent data bloat
    logger = setup_filtered_logging(args.log_dir, args.log_level)
    
    try:
        # Check and fix data_root path
        original_data_root = args.data_root
        args.data_root = find_data_directory(args.data_root)
        if args.data_root != original_data_root:
            logger.info(f"Updated data_root path from {original_data_root} to {args.data_root}")
        
        # Adjust output and log directories to be consistent with data_root
        if original_data_root != args.data_root:
            parent_dir = os.path.dirname(os.path.dirname(args.data_root))
            args.output_dir = os.path.join(parent_dir, "data", "tokenizer")
            args.log_dir = os.path.join(parent_dir, "outputs", "logs")
            logger.info(f"Updated output_dir to {args.output_dir}")
            logger.info(f"Updated log_dir to {args.log_dir}")
        
        # Load configuration if provided
        config = {}
        if os.path.exists(args.config):
            logger.info(f"Loading configuration from {args.config}")
            config = load_config(args.config)
        else:
            logger.info("No config file found, using command line arguments")
        
        # Parse sampling ratios
        sampling_ratios = parse_sampling_ratios(args.sampling_ratios)
        logger.info(f"Using sampling ratios: {sampling_ratios}")
        
        # Setup tokenizer training configuration
        tokenizer_config = {
            'model_type': args.model_type,
            'byte_level': args.byte_level,
            'vocab_size': args.vocab_size,
            'data_root': args.data_root,
            'output_dir': args.output_dir,
            'max_samples_per_dataset': args.max_samples,
            'sampling_ratios': sampling_ratios,
            'dataset_types': ['english', 'swedish', 'code'],
            'special_tokens': ["<s>", "</s>", "<unk>", "<pad>", "<mask>"]
        }
        
        # Update with any additional config from file
        for key, value in config.get('tokenizer', {}).items():
            if key not in tokenizer_config:
                tokenizer_config[key] = value
        
        # Create output directory
        os.makedirs(args.output_dir, exist_ok=True)
        
        # Initialize multilingual tokenizer trainer
        logger.info("Initializing multilingual tokenizer trainer...")
        trainer = MultilingualTokenizerTrainer(tokenizer_config)
        
        # Verify data directories exist
        for dataset_type in tokenizer_config['dataset_types']:
            data_dir = os.path.join(args.data_root, dataset_type)
            train_file = os.path.join(data_dir, "train.jsonl")
            
            if not os.path.exists(data_dir):
                logger.warning(f"Data directory {data_dir} does not exist")
                # List all subdirectories in data_root to help debug
                if os.path.exists(args.data_root):
                    subdirs = [d for d in os.listdir(args.data_root) if os.path.isdir(os.path.join(args.data_root, d))]
                    logger.info(f"Available subdirectories in {args.data_root}: {subdirs}")
            
            if not os.path.exists(train_file):
                logger.warning(f"Training file not found: {train_file}")
                # List all files in the directory to help debug
                if os.path.exists(data_dir):
                    files = os.listdir(data_dir)
                    logger.info(f"Files in {data_dir}: {files}")
        
        # Train tokenizer
        logger.info("Starting tokenizer training...")
        tokenizer = trainer.train_from_multilingual()
        
        # Save tokenizer
        logger.info("Saving tokenizer...")
        trainer.save_tokenizer(tokenizer)
        
        # Test the tokenizer with a simple example
        test_strings = [
            "This is an English test sentence.",
            "Detta är en svensk testmening.",
            "def hello_world():\n    print('Hello, world!')"
        ]
        
        logger.info("Testing tokenizer with sample texts:")
        loaded_tokenizer = trainer.load_tokenizer()
        
        for i, test_str in enumerate(test_strings):
            tokens = loaded_tokenizer.tokenize(test_str)
            token_count = len(tokens)
            # Only log a sample of tokens to avoid large logs
            sample_tokens = tokens[:10] + ['...'] + tokens[-5:] if token_count > 15 else tokens
            logger.info(f"Test {i+1} tokens ({token_count} total): {sample_tokens}")
        
        logger.info(f"Tokenizer training completed successfully! Saved to {args.output_dir}")
        
    except Exception as e:
        logger.error(f"Error in tokenizer training: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main() 