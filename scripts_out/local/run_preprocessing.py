#!/usr/bin/env python3
"""
Local execution script for data preprocessing.
This script provides a simple way to preprocess training data locally.
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
    from LiULLM.src.utils.config import load_config
    from LiULLM.src.data.preprocessing import TextPreprocessor
except ImportError:
    # If that fails, adjust path again to try direct import
    sys.path.insert(0, os.path.dirname(os.path.dirname(current_dir)))
    from src.utils.logging import setup_logging
    from src.utils.config import load_config
    from src.data.preprocessing import TextPreprocessor

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Preprocess data for LLM training locally")
    
    parser.add_argument(
        "--config", 
        type=str, 
        default="configs/data_config.yaml",
        help="Path to data configuration file"
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        default="data/raw",
        help="Directory containing raw data files"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/processed",
        help="Directory to save processed data"
    )
    parser.add_argument(
        "--train_file",
        type=str,
        default="train.jsonl",
        help="Filename for processed training data"
    )
    parser.add_argument(
        "--val_file",
        type=str,
        default="validation.jsonl",
        help="Filename for processed validation data"
    )
    parser.add_argument(
        "--val_split",
        type=float,
        default=0.05,
        help="Validation split ratio"
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
    
    logger.info("Starting data preprocessing")
    logger.info(f"Arguments: {args}")
    
    # Check if config file exists in current directory, if not look in LiULLM directory
    config_path = args.config
    if not os.path.exists(config_path):
        config_path = os.path.join(liullm_dir, args.config)
        logger.info(f"Config not found in current directory, trying: {config_path}")
    
    # Load configuration
    config = load_config(config_path)
    logger.info(f"Loaded configuration from: {config_path}")
    
    # Update config with CLI arguments if provided
    if not 'data' in config:
        config['data'] = {}
        
    if args.input_dir:
        config['data']['input_dir'] = args.input_dir
    if args.output_dir:
        config['data']['output_dir'] = args.output_dir
    if args.train_file:
        config['data']['train_file'] = args.train_file
    if args.val_file:
        config['data']['val_file'] = args.val_file
    if args.val_split:
        config['data']['val_split'] = args.val_split
    
    # Log the configuration
    logger.info(f"Data preprocessing configuration: {config['data']}")
    
    # Initialize text preprocessor
    preprocessor = TextPreprocessor(config)
    
    # Preprocess data
    logger.info("Starting data preprocessing...")
    train_samples, val_samples = preprocessor.preprocess_and_save()
    
    elapsed_time = time.time() - start_time
    logger.info(f"Data preprocessing completed in {elapsed_time:.2f} seconds")
    logger.info(f"Created {train_samples} training samples and {val_samples} validation samples")
    logger.info(f"Training data saved to: {os.path.join(args.output_dir, args.train_file)}")
    logger.info(f"Validation data saved to: {os.path.join(args.output_dir, args.val_file)}")

if __name__ == "__main__":
    main() 