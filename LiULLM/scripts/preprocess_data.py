#!/usr/bin/env python3
"""
Data preprocessing script for preparing the training corpus.
Processes raw multilingual text and creates train/validation splits.
"""

import os
import sys
import argparse
import logging
from typing import Dict, Any

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ""))

from src.utils.config import load_config, get_config_with_cli_overrides
from src.utils.logging import setup_logging, log_config
from src.data.preprocessing import TextPreprocessor

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Preprocess text data for LLM training")
    
    parser.add_argument(
        "--config", 
        type=str, 
        default="configs/data_config.yaml",
        help="Path to data configuration file"
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default=None,
        help="Directory to save processed data (overrides config)"
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
    """Main function to run data preprocessing."""
    # Parse arguments
    args = parse_args()
    
    # Set up logging
    log_level = logging.DEBUG if args.debug else logging.INFO
    logger = setup_logging(
        log_dir=args.log_dir,
        log_level=log_level,
        experiment_name="data_preprocessing"
    )
    
    try:
        # Load configuration
        logger.info(f"Loading configuration from {args.config}")
        config = load_config(args.config)
        
        # Override config with CLI arguments if provided
        cli_overrides = {}
        if args.output_dir:
            # Create processed_data dict if it doesn't exist
            if 'processed_data' not in config:
                config['processed_data'] = {}
            
            # Update output paths
            config['processed_data']['train'] = os.path.join(args.output_dir, "train.txt")
            config['processed_data']['validation'] = os.path.join(args.output_dir, "val.txt")
            
            # Ensure the directory exists
            os.makedirs(args.output_dir, exist_ok=True)
        
        # Log the configuration
        log_config(config)
        
        # Create and run the text preprocessor
        logger.info("Initializing text preprocessor")
        preprocessor = TextPreprocessor(config)
        
        # Process all languages and create train/val split
        logger.info("Starting data preprocessing")
        preprocessor.process_all_languages()
        
        logger.info("Data preprocessing completed successfully")
        
    except Exception as e:
        logger.error(f"Error in data preprocessing: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main() 