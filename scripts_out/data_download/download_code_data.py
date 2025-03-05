#!/usr/bin/env python3
"""
Dataset download script for the LiULLM pipeline.
Downloads code data from the EleutherAI Proof Pile 2 dataset.
"""

import os
import argparse
import logging
import sys
from pathlib import Path
import json
from tqdm import tqdm
from datasets import load_dataset
from concurrent.futures import ThreadPoolExecutor
import random

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), ""))

try:
    from src.utils.logging import setup_logging
except ImportError:
    # Fall back to relative import
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    from src.utils.logging import setup_logging

# Constants
DATASET_NAME = "EleutherAI/proof-pile-2"
DEFAULT_SIZE_LIMIT = 5 * 1024**3  # 5GB
BATCH_SIZE = 1000  # Number of examples to process at once
CHARS_PER_TOKEN = 3.6  # Approximation for calculating token count

logger = logging.getLogger(__name__)

def get_dataset_info():
    """
    Retrieve information about the dataset structure.
    """
    logger.info(f"Loading dataset info for {DATASET_NAME}")
    try:
        dataset_info = load_dataset(DATASET_NAME, streaming=True)
        return dataset_info
    except Exception as e:
        logger.error(f"Error loading dataset info: {e}")
        return None

def download_code_data(output_dir, languages=None, size_limit=DEFAULT_SIZE_LIMIT, 
                      max_samples=None, split="train"):
    """
    Download and process code data from the Proof Pile 2 dataset.
    
    Args:
        output_dir (str): Directory to save the processed data
        languages (list): List of programming languages to filter by (None for all)
        size_limit (int): Maximum size of downloaded data in bytes
        max_samples (int): Maximum number of samples to download (None for unlimited)
        split (str): Dataset split to use ('train', 'validation', etc.)
    
    Returns:
        list: Paths of saved data files
    """
    logger.info(f"Downloading code data from {DATASET_NAME}")
    os.makedirs(output_dir, exist_ok=True)
    
    # Get dataset
    try:
        dataset = load_dataset(DATASET_NAME, split=split, streaming=True)
    except Exception as e:
        logger.error(f"Error loading dataset: {e}")
        return []
    
    # Filter by language if specified
    if languages:
        logger.info(f"Filtering for languages: {languages}")
        # Note: Actual filtering depends on the exact structure of the dataset
        # This is a placeholder as we need to know the exact field name/structure
        # dataset = dataset.filter(lambda example: example.get('language', '') in languages)
    
    # Initialize counters and storage
    total_size = 0
    total_samples = 0
    saved_files = []
    file_index = 0
    
    # Process data in batches
    current_batch = []
    batch_size_bytes = 0
    
    logger.info(f"Starting data download with size limit: {size_limit/1024**2:.2f} MB")
    
    for example in tqdm(dataset, desc="Processing examples"):
        # Check if we've reached the sample limit
        if max_samples and total_samples >= max_samples:
            logger.info(f"Reached sample limit of {max_samples}")
            break
            
        # Process the example
        try:
            # Extract text (field name may vary depending on dataset structure)
            if 'text' in example:
                text = example['text']
            elif 'content' in example:
                text = example['content']
            else:
                # Try to find the main text field or convert the whole example to a string
                text = str(example)
            
            # Skip empty text
            if not text or text.isspace():
                continue
                
            # Calculate size
            example_size = len(text.encode('utf-8'))
            
            # Check if adding this example would exceed the size limit
            if total_size + example_size > size_limit:
                logger.info(f"Reached size limit of {size_limit/1024**2:.2f} MB")
                break
                
            # Add to current batch
            current_batch.append({
                'text': text,
                'tokens': len(text) // CHARS_PER_TOKEN,  # Rough estimate
                'meta': {
                    'source': 'proof-pile-2',
                    # Add any other metadata from the example
                    'id': example.get('id', str(total_samples))
                }
            })
            
            batch_size_bytes += example_size
            total_size += example_size
            total_samples += 1
            
            # Save batch when it reaches the target size
            if len(current_batch) >= BATCH_SIZE:
                file_path = save_batch(current_batch, output_dir, file_index)
                saved_files.append(file_path)
                file_index += 1
                current_batch = []
                batch_size_bytes = 0
                
        except Exception as e:
            logger.warning(f"Error processing example: {e}")
            continue
    
    # Save any remaining examples
    if current_batch:
        file_path = save_batch(current_batch, output_dir, file_index)
        saved_files.append(file_path)
    
    logger.info(f"Downloaded {total_samples} samples ({total_size/1024**2:.2f} MB) to {output_dir}")
    return saved_files

def save_batch(batch, output_dir, file_index):
    """
    Save a batch of examples to a file.
    
    Args:
        batch (list): List of examples to save
        output_dir (str): Directory to save the file
        file_index (int): Index for the file name
    
    Returns:
        str: Path to the saved file
    """
    file_path = os.path.join(output_dir, f"code_data_{file_index:04d}.jsonl")
    with open(file_path, 'w', encoding='utf-8') as f:
        for example in batch:
            f.write(json.dumps(example) + '\n')
    
    logger.debug(f"Saved {len(batch)} examples to {file_path}")
    return file_path

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Download code data from EleutherAI Proof Pile 2")
    
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/raw/code",
        help="Directory to save downloaded data"
    )
    parser.add_argument(
        "--languages",
        nargs="+",
        default=None,
        help="Programming languages to filter by (if supported by the dataset)"
    )
    parser.add_argument(
        "--size_limit_gb",
        type=float,
        default=DEFAULT_SIZE_LIMIT / 1024**3,
        help="Maximum size of data in GB"
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Maximum number of samples to download"
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        help="Dataset split to use (train, validation, etc.)"
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
    """Main function to download code data."""
    # Parse arguments
    args = parse_args()
    
    # Set up logging
    log_level = logging.DEBUG if args.debug else logging.INFO
    logger = setup_logging(
        log_dir=args.log_dir,
        log_level=log_level,
        experiment_name="code_data_download"
    )
    
    # Convert size limit to bytes
    size_limit = int(args.size_limit_gb * 1024**3)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Get dataset info first to understand the structure
    dataset_info = get_dataset_info()
    if not dataset_info:
        logger.error("Failed to get dataset info, aborting.")
        return
    
    # Download the data
    saved_files = download_code_data(
        output_dir=args.output_dir,
        languages=args.languages,
        size_limit=size_limit,
        max_samples=args.max_samples,
        split=args.split
    )
    
    # Print summary
    logger.info("Download Summary:")
    logger.info(f"Downloaded {len(saved_files)} files to {args.output_dir}")
    logger.info(f"Dataset: {DATASET_NAME}")
    if args.languages:
        logger.info(f"Languages: {', '.join(args.languages)}")
    
    logger.info("Download complete.")

if __name__ == "__main__":
    main() 