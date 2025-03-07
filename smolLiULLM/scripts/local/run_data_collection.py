#!/usr/bin/env python3
"""
Local execution script for data collection.
This script provides a simple way to download training data locally.
"""

import os
import sys
import argparse
import logging
import importlib.util
import time
from pathlib import Path

# Add smolLiULLM directory to path
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, root_dir)

# Import smolLiULLM modules using the Python import system
from src.utils.logging import setup_logging

def load_module_from_path(module_name, file_path):
    """Dynamically load a Python module from a file path."""
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Download data for LLM training locally")
    
    # General arguments
    parser.add_argument(
        "--data_type",
        type=str,
        choices=["pretraining", "instruction", "code"],
        default="pretraining",
        help="Type of data to download: pretraining, instruction, or code"
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
    
    # Pretraining data arguments
    parser.add_argument(
        "--languages",
        type=str,
        nargs="+",
        default=["en"],
        help="Languages to download (en, sv, is)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./data/raw",
        help="Directory to save downloaded data"
    )
    parser.add_argument(
        "--size_limit",
        type=float,
        default=5.0,
        help="Size limit for downloads in GB"
    )
    
    parser.add_argument(
        "--code_config",
        type=str,
        default="arxiv",
        choices=["arxiv", "math_stack", "open_web_math"],
        help="Configuration to use for code data download"
    )
    
    # Instruction data arguments
    parser.add_argument(
        "--dataset",
        type=str,
        default="yahma/alpaca-cleaned",
        help="Hugging Face dataset name for instruction data"
    )
    parser.add_argument(
        "--instruction_output_dir",
        type=str,
        default="./data/instruction_tuning",
        help="Directory to save instruction data"
    )
    parser.add_argument(
        "--train_filename",
        type=str,
        default="alpaca_cleaned_train.json",
        help="Filename for training instruction data"
    )
    parser.add_argument(
        "--val_filename",
        type=str,
        default="alpaca_cleaned_val.json",
        help="Filename for validation instruction data"
    )
    parser.add_argument(
        "--val_split",
        type=float,
        default=0.05,
        help="Validation split ratio for instruction data"
    )
    
    return parser.parse_args()

def setup_directories(directories):
    """Create directories if they don't exist."""
    for directory in directories:
        os.makedirs(directory, exist_ok=True)

def main():
    """Main function to run data collection."""
    start_time = time.time()
    args = parse_args()
    
    # Setup logging
    log_level = logging.DEBUG if args.debug else logging.INFO
    log_dir = args.log_dir
    timestamp = int(start_time)
    setup_logging(log_dir=log_dir, log_level=log_level, experiment_name=f"data_collection_{timestamp}")
    logger = logging.getLogger(__name__)
    
    # Create necessary directories
    setup_directories([args.log_dir, args.output_dir, args.instruction_output_dir])
    
    logger.info(f"Starting data collection in {args.data_type} mode")
    logger.info(f"Arguments: {args}")
    
    if args.data_type == "pretraining":
        # Import the download_datasets module
        download_datasets_path = os.path.join(root_dir, "scripts", "data_download", "download_datasets.py")
        download_datasets = load_module_from_path("download_datasets", download_datasets_path)
        
        # Download datasets for each language
        for lang in args.languages:
            logger.info(f"Downloading data for language: {lang}")
            if lang == "en":
                # Convert GB to bytes
                size_limit_bytes = int(args.size_limit * 1024**3)
                download_datasets.download_english_data(args.output_dir, size_limit_bytes)
            elif lang == "sv":
                # Swedish data requires min and max size in bytes
                min_bytes = int(args.size_limit * 0.8 * 1024**3)  # 80% of size_limit
                max_bytes = int(args.size_limit * 1024**3)
                download_datasets.download_swedish_data(args.output_dir, min_bytes, max_bytes)
            elif lang == "is":
                # Icelandic data also requires size limit
                size_limit_bytes = int(args.size_limit * 1024**3)
                download_datasets.download_icelandic_data(args.output_dir, size_limit_bytes)
            else:
                logger.warning(f"Unknown language: {lang}")
    
    elif args.data_type == "code":
        # Import the download_code_data module
        code_data_path = os.path.join(root_dir, "scripts", "data_download", "download_code_data.py")
        code_data = load_module_from_path("download_code_data", code_data_path)
        
        # Create code data directory
        code_output_dir = os.path.join(args.output_dir, "code")
        os.makedirs(code_output_dir, exist_ok=True)
        
        # Download code data
        logger.info("Downloading code data from Proof Pile 2")
        size_limit_bytes = int(args.size_limit * 1024**3)
        code_data.download_code_data(
            output_dir=code_output_dir,
            config=args.code_config,
            size_limit=size_limit_bytes
        )
    
    else:  # instruction data
        # Import the download_instruction_data module
        instruction_data_path = os.path.join(root_dir, "scripts", "data_download", "download_instruction_data.py")
        instruction_data = load_module_from_path("download_instruction_data", instruction_data_path)
        
        # Download instruction dataset
        logger.info(f"Downloading instruction data from {args.dataset}")
        instruction_data.download_instruction_dataset(
            dataset_name=args.dataset,
            output_dir=args.instruction_output_dir,
            train_filename=args.train_filename,
            val_filename=args.val_filename,
            val_split=args.val_split
        )
    
    elapsed_time = time.time() - start_time
    logger.info(f"Data collection completed in {elapsed_time:.2f} seconds")

if __name__ == "__main__":
    main() 