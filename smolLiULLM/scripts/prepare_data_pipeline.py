#!/usr/bin/env python3
"""
End-to-end data preparation pipeline that:
1. Downloads raw data from HuggingFace
2. Preprocesses it according to configuration
3. Prepares it for training
"""

import os
import sys
import argparse
import logging
import subprocess
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ""))

from src.utils.config import load_config
from src.utils.logging import setup_logging, log_config

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run the full data preparation pipeline")
    
    parser.add_argument(
        "--config", 
        type=str, 
        default="configs/data_config.yaml",
        help="Path to data configuration file"
    )
    parser.add_argument(
        "--skip_download", 
        action="store_true",
        help="Skip the download step (use existing raw data)"
    )
    parser.add_argument(
        "--skip_preprocessing", 
        action="store_true",
        help="Skip the preprocessing step"
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default=None,
        help="Directory to save processed data (overrides config)"
    )
    parser.add_argument(
        "--raw_dir", 
        type=str, 
        default=None,
        help="Directory with raw data (overrides config)"
    )
    parser.add_argument(
        "--log_dir", 
        type=str, 
        default="outputs/logs",
        help="Directory to save logs"
    )
    parser.add_argument(
        "--languages",
        nargs="+",
        choices=["english", "swedish", "icelandic", "all"],
        default=["all"],
        help="Languages to process"
    )
    parser.add_argument(
        "--debug", 
        action="store_true",
        help="Enable debug mode with more verbose logging"
    )
    
    return parser.parse_args()

def run_download(config, args, logger):
    """
    Run the data download step using the download_datasets.py script.
    """
    logger.info("Starting data download step")
    
    # Prepare command
    cmd = [
        sys.executable,
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "data_download", "download_datasets.py"),
        "--output_dir", args.raw_dir or os.path.dirname(config["raw_data"]["english"]),
        "--log_dir", args.log_dir
    ]
    
    # Add language arguments
    if args.languages and "all" not in args.languages:
        cmd.extend(["--languages"] + args.languages)
    
    # Add HuggingFace download configuration from config file
    if "huggingface_download" in config:
        hf_config = config["huggingface_download"]
        
        if "english" in hf_config:
            cmd.extend([
                "--english_size_gb", str(hf_config["english"]["size_gb"])
            ])
        
        if "swedish" in hf_config:
            cmd.extend([
                "--swedish_min_gb", str(hf_config["swedish"]["min_gb"]),
                "--swedish_max_gb", str(hf_config["swedish"]["max_gb"])
            ])
        
        if "icelandic" in hf_config:
            cmd.extend([
                "--icelandic_size_gb", str(hf_config["icelandic"]["size_gb"])
            ])
    
    # Add debug flag if needed
    if args.debug:
        cmd.append("--debug")
    
    # Run the download script
    logger.info(f"Running command: {' '.join(cmd)}")
    try:
        result = subprocess.run(cmd, check=True, text=True, capture_output=True)
        logger.info(f"Download output: {result.stdout}")
        if result.stderr:
            logger.warning(f"Download stderr: {result.stderr}")
        logger.info("Data download completed successfully")
    except subprocess.CalledProcessError as e:
        logger.error(f"Data download failed with exit code {e.returncode}")
        logger.error(f"Output: {e.stdout}")
        logger.error(f"Error: {e.stderr}")
        raise RuntimeError("Data download failed") from e

def run_preprocessing(config, args, logger):
    """
    Run the data preprocessing step using the preprocess_data.py script.
    """
    logger.info("Starting data preprocessing step")
    
    # Prepare command
    cmd = [
        sys.executable,
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "preprocess_data.py"),
        "--config", args.config,
        "--log_dir", args.log_dir
    ]
    
    # Add output directory if specified
    if args.output_dir:
        cmd.extend(["--output_dir", args.output_dir])
    
    # Add debug flag if needed
    if args.debug:
        cmd.append("--debug")
    
    # Run the preprocessing script
    logger.info(f"Running command: {' '.join(cmd)}")
    try:
        result = subprocess.run(cmd, check=True, text=True, capture_output=True)
        logger.info(f"Preprocessing output: {result.stdout}")
        if result.stderr:
            logger.warning(f"Preprocessing stderr: {result.stderr}")
        logger.info("Data preprocessing completed successfully")
    except subprocess.CalledProcessError as e:
        logger.error(f"Data preprocessing failed with exit code {e.returncode}")
        logger.error(f"Output: {e.stdout}")
        logger.error(f"Error: {e.stderr}")
        raise RuntimeError("Data preprocessing failed") from e

def main():
    """Main function to run the full data preparation pipeline."""
    # Parse arguments
    args = parse_args()
    
    # Set up logging
    log_level = logging.DEBUG if args.debug else logging.INFO
    logger = setup_logging(
        log_dir=args.log_dir,
        log_level=log_level,
        experiment_name="data_pipeline"
    )
    
    try:
        # Load configuration
        logger.info(f"Loading configuration from {args.config}")
        config = load_config(args.config)
        
        # Log the configuration
        log_config(config)
        
        # Run the download step if not skipped
        if not args.skip_download:
            run_download(config, args, logger)
        else:
            logger.info("Skipping download step as requested")
        
        # Run the preprocessing step if not skipped
        if not args.skip_preprocessing:
            run_preprocessing(config, args, logger)
        else:
            logger.info("Skipping preprocessing step as requested")
        
        logger.info("Data preparation pipeline completed successfully")
        
    except Exception as e:
        logger.error(f"Error in data preparation pipeline: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main() 