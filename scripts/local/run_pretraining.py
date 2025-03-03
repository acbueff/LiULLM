#!/usr/bin/env python3
"""
Local execution script for pretraining.
This script provides a simple way to run pretraining on a local GPU.
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
from src.models.llama_model import create_model, create_model_config, save_model
from src.training.trainer import train_model, set_seed

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run LLM pretraining locally")
    
    parser.add_argument(
        "--config", 
        type=str, 
        default="configs/pretrain_config.yaml",
        help="Path to pretraining configuration file"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data/processed",
        help="Directory containing processed data files"
    )
    parser.add_argument(
        "--tokenizer_path",
        type=str,
        default="data/tokenizer/tokenizer.json",
        help="Path to the tokenizer file"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/checkpoints/pretrain",
        help="Directory to save model checkpoints"
    )
    parser.add_argument(
        "--train_file",
        type=str,
        default="train.jsonl",
        help="Filename for training data"
    )
    parser.add_argument(
        "--val_file",
        type=str,
        default="validation.jsonl",
        help="Filename for validation data"
    )
    parser.add_argument(
        "--log_dir",
        type=str,
        default="outputs/logs",
        help="Directory to save logs"
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
        "--gradient_accumulation_steps",
        type=int,
        default=None,
        help="Gradient accumulation steps (overrides config)"
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="Path to checkpoint to resume training from"
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
    
    logger.info("Starting pretraining")
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
    if args.data_dir:
        config['data']['data_dir'] = args.data_dir
    if args.tokenizer_path:
        config['model']['tokenizer_path'] = args.tokenizer_path
    if args.output_dir:
        config['training']['output_dir'] = args.output_dir
    if args.train_file:
        config['data']['train_file'] = args.train_file
    if args.val_file:
        config['data']['val_file'] = args.val_file
    if args.num_epochs:
        config['training']['num_train_epochs'] = args.num_epochs
    if args.batch_size:
        config['training']['per_device_train_batch_size'] = args.batch_size
        config['training']['per_device_eval_batch_size'] = args.batch_size
    if args.gradient_accumulation_steps:
        config['training']['gradient_accumulation_steps'] = args.gradient_accumulation_steps
    if args.resume_from_checkpoint:
        config['training']['resume_from_checkpoint'] = args.resume_from_checkpoint
    
    # Set seed for reproducibility
    set_seed(config['training'].get('seed', 42))
    
    # Log the configuration
    logger.info(f"Training configuration: {config}")
    
    # Create model configuration
    model_config = create_model_config(config)
    
    # Create model
    model, tokenizer = create_model(model_config, config)
    
    # Log model information
    logger.info(f"Model created with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Train model
    logger.info("Starting model training...")
    train_stats = train_model(model, tokenizer, config)
    
    # Save final model
    final_model_path = os.path.join(args.output_dir, "final_model")
    save_model(model, tokenizer, final_model_path)
    
    elapsed_time = time.time() - start_time
    logger.info(f"Pretraining completed in {elapsed_time:.2f} seconds")
    logger.info(f"Final model saved to: {final_model_path}")
    logger.info(f"Training stats: {train_stats}")

if __name__ == "__main__":
    main() 