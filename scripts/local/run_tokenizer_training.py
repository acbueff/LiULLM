#!/usr/bin/env python3
"""
Local execution script for tokenizer training.
This script provides a simple way to train a tokenizer locally.
"""

import os
import sys
import argparse
import logging
import time
from pathlib import Path

# Add current directory to path to find LiULLM
current_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
liullm_dir = os.path.join(current_dir, "LiULLM")
sys.path.insert(0, liullm_dir)

# Now try to import from LiULLM
try:
    from LiULLM.src.utils.logging import setup_logging
except ImportError:
    # If that fails, adjust path again to try direct import
    sys.path.insert(0, os.path.dirname(os.path.dirname(current_dir)))
    from src.utils.logging import setup_logging
from src.utils.config import load_config
from src.data.tokenization import TokenizerTrainer

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train tokenizer for LLM locally")
    
    parser.add_argument(
        "--config", 
        type=str, 
        default="configs/tokenizer_config.yaml",
        help="Path to tokenizer configuration file"
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        default="data/processed",
        help="Directory containing processed data files"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/tokenizer",
        help="Directory to save tokenizer files"
    )
    parser.add_argument(
        "--vocab_size",
        type=int,
        default=32000,
        help="Size of the vocabulary"
    )
    parser.add_argument(
        "--sample_size",
        type=int,
        default=1000000,
        help="Number of samples to use for training"
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
        help="Enable debug mode for verbose logging"
    )
    
    return parser.parse_args()

def setup_directories(directories):
    """Create directories if they don't exist."""
    for directory in directories:
        os.makedirs(directory, exist_ok=True)

def main():
    """Main function to run the script."""
    # Get script name without extension for logging
    script_name = os.path.basename(__file__).replace(".py", "")
    start_time = time.time()
    args = parse_args()
    
    # Setup logging - using the correct parameters for the setup_logging function
    
    log_level = logging.DEBUG if args.debug else logging.INFO
    
    log_timestamp = int(start_time)
    
    experiment_name = f"{script_name}_{log_timestamp}"
    
    
    
    # Create the log directory if it doesn\'t exist
    
    os.makedirs(args.log_dir, exist_ok=True)
    
    
    
    # Setup logging with the correct parameters
    
    setup_logging(
    
        log_dir=args.log_dir,
    
        log_level=log_level,
    
        experiment_name=experiment_name
    
    )
    logger = logging.getLogger(__name__)
    
    # Create necessary directories
    setup_directories([args.log_dir, args.output_dir])
    
    logger.info("Starting tokenizer training")
    logger.info(f"Arguments: {args}")
    
    # Load configuration
    # Check if config file exists in current directory, if not look in LiULLM directory
    config_path = args.config
    if not os.path.exists(config_path):
        config_path = os.path.join(liullm_dir, args.config)
        logger.info(f"Config not found in current directory, trying: {config_path}")
    
    # Load configuration
    config = load_config(config_path)
    logger.info(f"Loaded configuration from: {config_path}")
    
    # Update config with CLI arguments if provided
    if args.input_dir:
        config['tokenizer']['input_dir'] = args.input_dir
    if args.output_dir:
        config['tokenizer']['output_dir'] = args.output_dir
    if args.vocab_size:
        config['tokenizer']['vocab_size'] = args.vocab_size
    if args.sample_size:
        config['tokenizer']['sample_size'] = args.sample_size
    
    # Log the configuration
    logger.info(f"Tokenizer configuration: {config['tokenizer']}")
    
    # Initialize tokenizer trainer
    tokenizer_trainer = TokenizerTrainer(config)
    
    # Train tokenizer
    logger.info("Starting tokenizer training...")
    tokenizer = tokenizer_trainer.train_tokenizer()
    
    # Save tokenizer
    tokenizer_path = os.path.join(args.output_dir, "tokenizer.json")
    tokenizer.save(tokenizer_path)
    
    elapsed_time = time.time() - start_time
    logger.info(f"Tokenizer training completed in {elapsed_time:.2f} seconds")
    logger.info(f"Tokenizer saved to: {tokenizer_path}")
    logger.info(f"Vocabulary size: {tokenizer.get_vocab_size()}")

if __name__ == "__main__":
    main() 