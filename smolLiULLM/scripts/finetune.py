#!/usr/bin/env python3
"""
Fine-tuning script for instruction-tuning a pretrained language model.
Supports Parameter-Efficient Fine-Tuning (PEFT) methods like LoRA.
"""

import os
import sys
import argparse
import logging
import torch
from typing import Dict, Any

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ""))

from src.utils.config import load_config
from src.utils.logging import setup_logging, log_config
from src.models.llama_model import load_model
from src.training.finetuning import finetune_model, setup_peft
from src.training.trainer import set_seed

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Fine-tune language model with instructions")
    
    parser.add_argument(
        "--config", 
        type=str, 
        default="configs/finetune_config.yaml",
        help="Path to fine-tuning configuration file"
    )
    parser.add_argument(
        "--model_path", 
        type=str, 
        default=None,
        help="Path to pretrained model (overrides config)"
    )
    parser.add_argument(
        "--tokenizer_path", 
        type=str, 
        default=None,
        help="Path to tokenizer (overrides config)"
    )
    parser.add_argument(
        "--data_path", 
        type=str, 
        default=None,
        help="Path to instruction data (overrides config)"
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default=None,
        help="Directory to save fine-tuned model (overrides config)"
    )
    parser.add_argument(
        "--lora_r", 
        type=int, 
        default=None,
        help="LoRA r dimension (overrides config)"
    )
    parser.add_argument(
        "--lora_alpha", 
        type=int, 
        default=None,
        help="LoRA alpha parameter (overrides config)"
    )
    parser.add_argument(
        "--lora_dropout", 
        type=float, 
        default=None,
        help="LoRA dropout rate (overrides config)"
    )
    parser.add_argument(
        "--local_rank", 
        type=int, 
        default=-1,
        help="Local rank for distributed training"
    )
    parser.add_argument(
        "--seed", 
        type=int, 
        default=None,
        help="Random seed (overrides config)"
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
    """Main function to run fine-tuning."""
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
            experiment_name="finetuning"
        )
    else:
        logger = logging.getLogger()
        logger.setLevel(logging.WARNING)
    
    try:
        # Load configuration
        logger.info(f"Loading fine-tuning configuration from {args.config}")
        config = load_config(args.config)
        
        # Override config with CLI arguments if provided
        if args.model_path:
            config['model_path'] = args.model_path
            logger.info(f"Using model from: {args.model_path}")
        
        if args.tokenizer_path:
            config['tokenizer_path'] = args.tokenizer_path
            logger.info(f"Using tokenizer from: {args.tokenizer_path}")
        
        if args.data_path:
            config['data_path'] = args.data_path
            logger.info(f"Using instruction data from: {args.data_path}")
        
        if args.output_dir:
            config['output_dir'] = args.output_dir
            os.makedirs(args.output_dir, exist_ok=True)
            logger.info(f"Using output directory: {args.output_dir}")
        
        if args.seed is not None:
            config['seed'] = args.seed
            logger.info(f"Using random seed: {args.seed}")
        
        # LoRA configuration overrides
        if args.lora_r is not None:
            config['peft_config']['r'] = args.lora_r
            logger.info(f"Using LoRA r: {args.lora_r}")
        
        if args.lora_alpha is not None:
            config['peft_config']['lora_alpha'] = args.lora_alpha
            logger.info(f"Using LoRA alpha: {args.lora_alpha}")
        
        if args.lora_dropout is not None:
            config['peft_config']['lora_dropout'] = args.lora_dropout
            logger.info(f"Using LoRA dropout: {args.lora_dropout}")
        
        config['use_wandb'] = args.wandb
        if args.wandb:
            logger.info("Weights & Biases logging enabled")
        
        # Log config if main process
        if is_main_process:
            log_config(config)
        
        # Set random seed for reproducibility
        set_seed(config.get('seed', 42))
        
        # Load base model
        logger.info(f"Loading pretrained model from {config['model_path']}")
        model, tokenizer = load_model(
            model_dir=config['model_path'],
            tokenizer_path=config['tokenizer_path'],
            local_rank=args.local_rank
        )
        
        # Fine-tune model
        logger.info("Starting fine-tuning...")
        finetune_model(
            model=model,
            tokenizer=tokenizer,
            config=config,
            local_rank=args.local_rank
        )
        
        if is_main_process:
            logger.info("Fine-tuning completed successfully!")
        
    except Exception as e:
        logger.error(f"Error in fine-tuning: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main() 