#!/usr/bin/env python3
"""
Tokenizer training script for creating a subword tokenizer
(byte-level BPE) for multilingual text.
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
from src.data.tokenization import TokenizerTrainer

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train tokenizer for LLM")
    
    parser.add_argument(
        "--config", 
        type=str, 
        default="configs/tokenizer_config.yaml",
        help="Path to tokenizer configuration file"
    )
    parser.add_argument(
        "--input_file", 
        type=str, 
        default=None,
        help="Path to input training data file (overrides config)"
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default=None,
        help="Directory to save tokenizer (overrides config)"
    )
    parser.add_argument(
        "--vocab_size", 
        type=int, 
        default=None,
        help="Vocabulary size (overrides config)"
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
    """Main function to run tokenizer training."""
    # Parse arguments
    args = parse_args()
    
    # Set up logging
    log_level = logging.DEBUG if args.debug else logging.INFO
    logger = setup_logging(
        log_dir=args.log_dir,
        log_level=log_level,
        experiment_name="tokenizer_training"
    )
    
    try:
        # Load configuration
        logger.info(f"Loading configuration from {args.config}")
        config = load_config(args.config)
        
        # Override config with CLI arguments if provided
        if args.input_file:
            config['train_data'] = args.input_file
            logger.info(f"Using input file: {args.input_file}")
            
        if args.output_dir:
            config['output_dir'] = args.output_dir
            # Ensure the directory exists
            os.makedirs(args.output_dir, exist_ok=True)
            logger.info(f"Using output directory: {args.output_dir}")
        
        if args.vocab_size:
            config['vocab_size'] = args.vocab_size
            logger.info(f"Using vocabulary size: {args.vocab_size}")
        
        # Log the configuration
        log_config(config)
        
        # Verify input file exists
        if not os.path.exists(config['train_data']):
            logger.error(f"Input file not found: {config['train_data']}")
            sys.exit(1)
        
        # Create tokenizer trainer
        logger.info("Initializing tokenizer trainer")
        tokenizer_trainer = TokenizerTrainer(config)
        
        # Train and save tokenizer
        logger.info("Starting tokenizer training")
        tokenizer_trainer.train_and_save()
        
        logger.info(f"Tokenizer training completed successfully, saved to {config['output_dir']}")
        
        # Load and test the tokenizer
        logger.info("Loading and testing the trained tokenizer")
        tokenizer = tokenizer_trainer.load_tokenizer()
        
        # Test tokenization on a sample text
        sample_text = "Testing the multilingual tokenizer with some text."
        tokens = tokenizer.tokenize(sample_text)
        logger.info(f"Sample tokenization: '{sample_text}' → {tokens}")
        
    except Exception as e:
        logger.error(f"Error in tokenizer training: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main() 