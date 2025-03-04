#!/usr/bin/env python3
"""
Script to run instruction fine-tuning on a pretrained model.
Uses the downloaded instruction tuning data.
"""

import os
import sys
import argparse
import logging
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ""))

from src.utils.config import load_config
from src.utils.logging import setup_logging, log_config
from src.training.finetuning import prepare_instruction_dataset, finetune_model
from transformers import AutoModelForCausalLM, AutoTokenizer

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run instruction fine-tuning")
    
    parser.add_argument(
        "--config", 
        type=str, 
        default="configs/finetune_config.yaml",
        help="Path to fine-tuning configuration file"
    )
    parser.add_argument(
        "--model_path", 
        type=str, 
        required=True,
        help="Path to pretrained model to fine-tune"
    )
    parser.add_argument(
        "--train_file", 
        type=str, 
        default="data/instruction_tuning/alpaca_cleaned_train.json",
        help="Path to instruction training data file"
    )
    parser.add_argument(
        "--val_file", 
        type=str, 
        default="data/instruction_tuning/alpaca_cleaned_val.json",
        help="Path to instruction validation data file"
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default="outputs/finetuned",
        help="Directory to save fine-tuned model"
    )
    parser.add_argument(
        "--log_dir", 
        type=str, 
        default="outputs/logs",
        help="Directory to save logs"
    )
    parser.add_argument(
        "--prompt_template", 
        type=str, 
        default="<s>User: {instruction}\nAssistant: ",
        help="Template for user instruction prompt"
    )
    parser.add_argument(
        "--completion_template", 
        type=str, 
        default="{response}</s>",
        help="Template for assistant response"
    )
    parser.add_argument(
        "--max_seq_length",
        type=int,
        default=2048,
        help="Maximum sequence length"
    )
    parser.add_argument(
        "--debug", 
        action="store_true",
        help="Enable debug mode with more verbose logging"
    )
    
    return parser.parse_args()

def main():
    """Main function to run instruction fine-tuning."""
    # Parse arguments
    args = parse_args()
    
    # Set up logging
    log_level = logging.DEBUG if args.debug else logging.INFO
    logger = setup_logging(
        log_dir=args.log_dir,
        log_level=log_level,
        experiment_name="instruction_finetuning"
    )
    
    try:
        # Load configuration
        logger.info(f"Loading configuration from {args.config}")
        config = load_config(args.config)
        
        # Override config with CLI arguments if provided
        if args.output_dir:
            config["output_dir"] = args.output_dir
        
        # Create output directory
        os.makedirs(config["output_dir"], exist_ok=True)
        
        # Log the configuration
        log_config(config)
        
        # Load model and tokenizer
        logger.info(f"Loading model and tokenizer from {args.model_path}")
        tokenizer = AutoTokenizer.from_pretrained(args.model_path)
        model = AutoModelForCausalLM.from_pretrained(args.model_path)
        
        # Prepare instruction datasets
        logger.info(f"Preparing instruction datasets from {args.train_file} and {args.val_file}")
        train_dataset, validation_dataset = prepare_instruction_dataset(
            tokenizer=tokenizer,
            train_file=args.train_file,
            validation_file=args.val_file,
            max_seq_length=args.max_seq_length,
            prompt_template=args.prompt_template,
            completion_template=args.completion_template
        )
        
        # Run fine-tuning
        logger.info("Starting fine-tuning")
        model, metrics = finetune_model(
            model=model,
            tokenizer=tokenizer,
            train_dataset=train_dataset,
            validation_dataset=validation_dataset,
            config=config,
            output_dir=config["output_dir"]
        )
        
        logger.info(f"Fine-tuning completed with metrics: {metrics}")
        logger.info(f"Model saved to {config['output_dir']}")
        
    except Exception as e:
        logger.error(f"Error in instruction fine-tuning: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main() 