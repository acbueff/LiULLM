#!/usr/bin/env python3
"""
Local execution script for fine-tuning.
This script provides a simple way to run fine-tuning on a local GPU.
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
from src.models.llama_model import load_model, save_model
from src.training.trainer import train_model, set_seed

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run LLM fine-tuning locally")
    
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
        help="Path to the pretrained model"
    )
    parser.add_argument(
        "--train_file",
        type=str,
        default="data/instruction_tuning/alpaca_cleaned_train.json",
        help="Path to the instruction training data file"
    )
    parser.add_argument(
        "--val_file",
        type=str,
        default="data/instruction_tuning/alpaca_cleaned_val.json",
        help="Path to the instruction validation data file"
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
        "--num_epochs",
        type=int,
        default=None,
        help="Number of training epochs (overrides config)"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="Batch size for training (overrides config)"
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=None,
        help="Learning rate (overrides config)"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode for verbose logging"
    )
    parser.add_argument(
        "--gpu_id",
        type=int,
        default=0,
        help="GPU ID to use for training (single GPU)"
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
    
    logger.info("Starting fine-tuning")
    logger.info(f"Arguments: {args}")
    
    # Set specific GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    
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
    if args.output_dir:
        config['output_dir'] = args.output_dir
    
    # Update training parameters if provided
    if args.num_epochs:
        config['training']['num_train_epochs'] = args.num_epochs
    if args.batch_size:
        config['training']['per_device_train_batch_size'] = args.batch_size
        config['training']['per_device_eval_batch_size'] = args.batch_size
    if args.learning_rate:
        config['training']['learning_rate'] = args.learning_rate
    
    # Update data configuration if provided
    if not 'data' in config:
        config['data'] = {}
    config['data']['prompt_template'] = args.prompt_template
    config['data']['completion_template'] = args.completion_template
    config['data']['max_length'] = args.max_seq_length
    
    # Set seed for reproducibility
    set_seed(config['training'].get('seed', 42))
    
    # Log the configuration
    logger.info(f"Fine-tuning configuration: {config}")
    
    # Load model and tokenizer
    logger.info(f"Loading model from: {args.model_path}")
    model, tokenizer = load_model(args.model_path)
    
    # Log model information
    logger.info(f"Model loaded with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Configure datasets
    config['data']['train_file'] = args.train_file
    config['data']['val_file'] = args.val_file
    
    # Train model (fine-tuning)
    logger.info("Starting model fine-tuning...")
    train_stats = train_model(model, tokenizer, config, is_finetune=True)
    
    # Save final fine-tuned model
    final_model_path = os.path.join(args.output_dir, "final_model")
    save_model(model, tokenizer, final_model_path)
    
    elapsed_time = time.time() - start_time
    logger.info(f"Fine-tuning completed in {elapsed_time:.2f} seconds")
    logger.info(f"Fine-tuned model saved to: {final_model_path}")
    logger.info(f"Training stats: {train_stats}")

if __name__ == "__main__":
    main() 