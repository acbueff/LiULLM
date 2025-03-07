#!/usr/bin/env python3
"""
Full training pipeline for LiULLM: tokenizer training followed by multilingual pretraining.
This script orchestrates the complete process of:
1. Training a tokenizer on the multilingual data
2. Analyzing the data and applying the Chinchilla scaling law
3. Pretraining the model across English, Swedish, and code datasets
"""

import os
import sys
import argparse
import logging
import subprocess
import yaml
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ""))

from src.utils.logging import setup_logging
from src.utils.config import load_config

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run the full LiULLM training pipeline")
    
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
        "--tokenizer_config", 
        type=str, 
        default="configs/tokenizer_config.yaml",
        help="Path to tokenizer configuration file"
    )
    parser.add_argument(
        "--data_root", 
        type=str, 
        default="data/processed",
        help="Root directory with training data subdirectories"
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default="outputs/liullm",
        help="Base output directory for all artifacts"
    )
    parser.add_argument(
        "--tokenizer_vocab_size", 
        type=int, 
        default=32000,
        help="Vocabulary size for the tokenizer"
    )
    parser.add_argument(
        "--samples_per_type", 
        type=str, 
        default="english=0.4,swedish=0.4,code=0.2",
        help="Sampling ratio for each dataset type for tokenizer training"
    )
    parser.add_argument(
        "--skip_tokenizer", 
        action="store_true",
        help="Skip tokenizer training and use existing tokenizer"
    )
    parser.add_argument(
        "--skip_analysis", 
        action="store_true",
        help="Skip data analysis and model scaling calculation"
    )
    parser.add_argument(
        "--skip_pretraining", 
        action="store_true",
        help="Skip model pretraining"
    )
    parser.add_argument(
        "--resume_from_checkpoint", 
        type=str, 
        default=None,
        help="Path to checkpoint to resume training from"
    )
    parser.add_argument(
        "--wandb_project", 
        type=str, 
        default="multilingual-llm",
        help="Weights & Biases project name"
    )
    parser.add_argument(
        "--wandb_entity", 
        type=str, 
        default=None,
        help="Weights & Biases entity name"
    )
    parser.add_argument(
        "--seed", 
        type=int, 
        default=42,
        help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--debug", 
        action="store_true",
        help="Enable debug mode with more verbose logging"
    )
    
    return parser.parse_args()

def setup_directories(args):
    """Set up directory structure and return paths."""
    # Create base directories
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set up specific directories
    paths = {
        'tokenizer_dir': os.path.join(args.output_dir, "tokenizer"),
        'model_dir': os.path.join(args.output_dir, "model"),
        'log_dir': os.path.join(args.output_dir, "logs"),
        'checkpoint_dir': os.path.join(args.output_dir, "checkpoints"),
        'config_dir': os.path.join(args.output_dir, "configs")
    }
    
    # Create all directories
    for dir_path in paths.values():
        os.makedirs(dir_path, exist_ok=True)
    
    return paths

def train_tokenizer(args, paths, logger):
    """Run the multilingual tokenizer training script."""
    logger.info("Starting multilingual tokenizer training...")
    
    # Construct command
    cmd = [
        "python", 
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "train_multilingual_tokenizer.py"),
        "--config", args.tokenizer_config,
        "--data_root", args.data_root,
        "--output_dir", paths['tokenizer_dir'],
        "--vocab_size", str(args.tokenizer_vocab_size),
        "--samples_per_type", args.samples_per_type,
        "--log_dir", paths['log_dir']
    ]
    
    if args.debug:
        cmd.append("--debug")
    
    # Run the tokenizer training process
    logger.info(f"Running command: {' '.join(cmd)}")
    process = subprocess.run(cmd, check=True)
    
    if process.returncode != 0:
        logger.error("Tokenizer training failed!")
        sys.exit(1)
    
    logger.info("Tokenizer training completed successfully")
    return os.path.join(paths['tokenizer_dir'], "tokenizer.json")

def analyze_data(args, paths, tokenizer_path, logger):
    """Run data analysis to determine optimal model scaling."""
    logger.info("Starting data analysis for Chinchilla scaling...")
    
    # Output path for scaling recommendations
    scaling_file = os.path.join(paths['config_dir'], "model_scaling.yaml")
    
    # Extract dataset types from the sampling ratios string
    dataset_types = [part.split('=')[0] for part in args.samples_per_type.split(',')]
    
    # Verify tokenizer exists before proceeding
    if not os.path.exists(tokenizer_path):
        logger.error(f"Tokenizer file not found at {tokenizer_path}")
        logger.error("Make sure to train the tokenizer first using --skip_tokenizer=False")
        sys.exit(1)
    
    # Get the absolute path to the analyze_data_for_scaling.py script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    script_path = os.path.join(script_dir, "analyze_data_for_scaling.py")
    
    # Ensure script path exists and is accessible
    if not os.path.exists(script_path):
        logger.error(f"Analysis script not found at {script_path}")
        # Try to find the script in alternative locations
        alt_path = os.path.join(os.path.dirname(script_dir), "scripts", "analyze_data_for_scaling.py")
        if os.path.exists(alt_path):
            logger.info(f"Found script at alternative path: {alt_path}")
            script_path = alt_path
        else:
            logger.error("Could not find analyze_data_for_scaling.py script")
            sys.exit(1)
    
    # Construct command with tokenizer_file directly specified
    cmd = [
        "python", 
        script_path,
        "--data_root", os.path.abspath(args.data_root),
        "--tokenizer_path", os.path.abspath(tokenizer_path),
        "--output_file", scaling_file,
        "--dataset_types", ",".join(dataset_types),
        "--log_dir", paths['log_dir']
    ]
    
    if args.debug:
        cmd.append("--debug")
    
    # Log the full command to verify the path
    logger.info(f"Running command: {' '.join(cmd)}")
    logger.info(f"Script path: {script_path}")
    logger.info(f"Tokenizer path: {os.path.abspath(tokenizer_path)}")
    
    # Run the analysis process
    process = subprocess.run(cmd, check=True)
    
    if process.returncode != 0:
        logger.error("Data analysis failed!")
        sys.exit(1)
    
    logger.info(f"Data analysis completed successfully, recommendations saved to {scaling_file}")
    return scaling_file

def update_model_config(model_config_path, scaling_file, output_path, logger):
    """Update model configuration based on scaling recommendations."""
    logger.info(f"Updating model configuration based on scaling recommendations...")
    
    # Load original model config
    with open(model_config_path, 'r') as f:
        model_config = yaml.safe_load(f)
    
    # Load scaling recommendations
    with open(scaling_file, 'r') as f:
        scaling = yaml.safe_load(f)
    
    # Update model architecture parameters
    model_config['model_arch'] = scaling['model_architecture']
    
    # Update training recommendations
    if 'training' not in model_config:
        model_config['training'] = {}
    
    for k, v in scaling['training_recommendations'].items():
        model_config['training'][k] = v
    
    # Add data analysis information
    model_config['data_analysis'] = scaling['data_analysis']
    
    # Save updated config
    with open(output_path, 'w') as f:
        yaml.safe_dump(model_config, f, sort_keys=False)
    
    logger.info(f"Updated model configuration saved to {output_path}")
    return output_path

def run_pretraining(args, paths, tokenizer_path, model_config_path, logger):
    """Run the multilingual pretraining script."""
    logger.info("Starting multilingual pretraining...")
    
    # Construct command
    cmd = [
        "python", 
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "pretrain_multilingual.py"),
        "--config", args.config,
        "--model_config", model_config_path,
        "--data_root", args.data_root,
        "--output_dir", paths['model_dir'],
        "--tokenizer_path", paths['tokenizer_dir'],
        "--log_dir", paths['log_dir'],
        "--wandb_project", args.wandb_project,
        "--seed", str(args.seed)
    ]
    
    if args.wandb_entity:
        cmd.extend(["--wandb_entity", args.wandb_entity])
    
    if args.resume_from_checkpoint:
        cmd.extend(["--resume_from_checkpoint", args.resume_from_checkpoint])
    
    if args.debug:
        cmd.append("--debug")
    
    # Run the pretraining process
    logger.info(f"Running command: {' '.join(cmd)}")
    process = subprocess.run(cmd, check=True)
    
    if process.returncode != 0:
        logger.error("Model pretraining failed!")
        sys.exit(1)
    
    logger.info("Model pretraining completed successfully")

def main():
    """Main function to run the full training pipeline."""
    # Parse arguments
    args = parse_args()
    
    # Set up directories
    paths = setup_directories(args)
    
    # Set up logging
    log_level = logging.DEBUG if args.debug else logging.INFO
    logger = setup_logging(
        log_dir=paths['log_dir'],
        log_level=log_level,
        experiment_name="full_training_pipeline"
    )
    
    logger.info("Starting LiULLM full training pipeline")
    logger.info(f"Using data from: {args.data_root}")
    logger.info(f"Output directory: {args.output_dir}")
    
    try:
        # Step 1: Train tokenizer (unless skipped)
        tokenizer_path = os.path.join(paths['tokenizer_dir'], "tokenizer.json")
        
        if not args.skip_tokenizer:
            logger.info("Starting tokenizer training...")
            tokenizer_path = train_tokenizer(args, paths, logger)
        else:
            logger.info(f"Checking for existing tokenizer at {tokenizer_path}")
            if not os.path.exists(tokenizer_path):
                # Search common locations for tokenizer files
                possible_locations = [
                    os.path.join(paths['tokenizer_dir'], "tokenizer.json"),
                    os.path.join("data", "tokenizer", "tokenizer.json"),
                    os.path.join("LiULLM", "smolLiULLM", "data", "tokenizer", "tokenizer.json"),
                ]
                
                for loc in possible_locations:
                    if os.path.exists(loc):
                        tokenizer_path = loc
                        logger.info(f"Found tokenizer at {tokenizer_path}")
                        break
                
                if not os.path.exists(tokenizer_path):
                    logger.error("Tokenizer not found! Cannot proceed with data analysis.")
                    logger.error("Please train a tokenizer first by removing --skip_tokenizer flag.")
                    sys.exit(1)
        
        # Step 2: Analyze data and determine model scaling (unless skipped)
        if not args.skip_analysis:
            logger.info("Starting data analysis with tokenizer from: " + tokenizer_path)
            scaling_file = analyze_data(args, paths, tokenizer_path, logger)
            
            # Update model config based on scaling recommendations
            updated_model_config = os.path.join(paths['config_dir'], "model_config_scaled.yaml")
            model_config_path = update_model_config(
                args.model_config,
                scaling_file,
                updated_model_config,
                logger
            )
        else:
            logger.info("Skipping data analysis and model scaling")
            model_config_path = args.model_config
        
        # Step 3: Run pretraining (unless skipped)
        if not args.skip_pretraining:
            run_pretraining(args, paths, tokenizer_path, model_config_path, logger)
        else:
            logger.info("Skipping model pretraining as requested")
        
        logger.info("LiULLM training pipeline completed successfully!")
        
    except Exception as e:
        logger.error(f"Error in training pipeline: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()