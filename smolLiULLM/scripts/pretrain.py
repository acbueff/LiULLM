#!/usr/bin/env python3
"""
Pretraining script for training a language model from scratch or 
continuing training from a checkpoint.
"""

import os
import sys
import argparse
import logging
import torch
from typing import Dict, Any

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ""))

from src.utils.config import load_config, get_config_with_cli_overrides
from src.utils.logging import setup_logging, log_config
from src.models.llama_model import create_model, create_model_config, save_model
from src.training.trainer import train_model, set_seed

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Pretrain language model")
    
    parser.add_argument(
        "--config", 
        type=str, 
        default="configs/pretrain_config.yaml",
        help="Path to pretraining configuration file"
    )
    parser.add_argument(
        "--model_config", 
        type=str, 
        default="configs/model_config.yaml",
        help="Path to model configuration file"
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default=None,
        help="Directory to save model checkpoints (overrides config)"
    )
    parser.add_argument(
        "--data_dir", 
        type=str, 
        default=None,
        help="Directory with training data (overrides config)"
    )
    parser.add_argument(
        "--tokenizer_path", 
        type=str, 
        default=None,
        help="Path to tokenizer (overrides config)"
    )
    parser.add_argument(
        "--resume_from_checkpoint", 
        type=str, 
        default=None,
        help="Path to checkpoint to resume training from (overrides config)"
    )
    parser.add_argument(
        "--seed", 
        type=int, 
        default=None,
        help="Random seed (overrides config)"
    )
    parser.add_argument(
        "--local_rank", 
        type=int, 
        default=-1,
        help="Local rank for distributed training"
    )
    parser.add_argument(
        "--log_dir", 
        type=str, 
        default="outputs/logs",
        help="Directory to save logs"
    )
    parser.add_argument(
        "--wandb", 
        action="store_true",
        help="Enable Weights & Biases logging"
    )
    parser.add_argument(
        "--debug", 
        action="store_true",
        help="Enable debug mode with more verbose logging"
    )
    
    return parser.parse_args()

def main():
    """Main function to run pretraining."""
    # Parse arguments
    args = parse_args()
    
    # Set up distributed training if needed
    if args.local_rank != -1:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend="nccl")
        is_main_process = args.local_rank == 0
    else:
        is_main_process = True
    
    # Set up logging (only on main process)
    if is_main_process:
        log_level = logging.DEBUG if args.debug else logging.INFO
        logger = setup_logging(
            log_dir=args.log_dir,
            log_level=log_level,
            experiment_name="pretraining"
        )
    else:
        logger = logging.getLogger()
        logger.setLevel(logging.WARNING)
    
    try:
        # Load configurations
        logger.info(f"Loading pretraining configuration from {args.config}")
        train_config = load_config(args.config)
        
        logger.info(f"Loading model configuration from {args.model_config}")
        model_config = load_config(args.model_config)
        
        # Override configs with CLI arguments if provided
        if args.output_dir:
            train_config['output_dir'] = args.output_dir
            os.makedirs(args.output_dir, exist_ok=True)
            logger.info(f"Using output directory: {args.output_dir}")
        
        if args.data_dir:
            train_config['data_dir'] = args.data_dir
            logger.info(f"Using data directory: {args.data_dir}")
        
        if args.tokenizer_path:
            train_config['tokenizer_path'] = args.tokenizer_path
            logger.info(f"Using tokenizer from: {args.tokenizer_path}")
        
        if args.resume_from_checkpoint:
            train_config['resume_from_checkpoint'] = args.resume_from_checkpoint
            logger.info(f"Resuming from checkpoint: {args.resume_from_checkpoint}")
        
        if args.seed is not None:
            train_config['seed'] = args.seed
            logger.info(f"Using random seed: {args.seed}")
        
        train_config['use_wandb'] = args.wandb
        if args.wandb:
            logger.info("Weights & Biases logging enabled")
        
        # Set random seed for reproducibility
        set_seed(train_config.get('seed', 42))
        
        # Log configurations if main process
        if is_main_process:
            log_config(train_config, "Training Configuration")
            log_config(model_config, "Model Configuration")
        
        # Create model
        logger.info("Creating model...")
        model_configuration = create_model_config(model_config)
        model = create_model(
            config=model_configuration,
            pretrained=train_config.get('resume_from_checkpoint', None)
        )
        
        # Train model
        logger.info("Starting model training...")
        trained_model = train_model(
            model=model,
            train_config=train_config,
            model_config=model_config,
            local_rank=args.local_rank
        )
        
        # Save final model if main process
        if is_main_process:
            logger.info("Saving final model...")
            save_model(
                model=trained_model,
                output_dir=os.path.join(train_config['output_dir'], "final"),
                tokenizer_path=train_config['tokenizer_path']
            )
            
            logger.info("Pretraining completed successfully!")
        
    except Exception as e:
        logger.error(f"Error in pretraining: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main() 