#!/usr/bin/env python3
"""
Script for training a multilingual tokenizer using data from multiple languages
and domains (English, Swedish, and code) for use with the LiULLM model.
"""

import os
import sys
import argparse
import logging
import json
import random
from typing import Dict, List, Any
from pathlib import Path
from tqdm import tqdm

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ""))

from src.utils.logging import setup_logging
from src.utils.config import load_config

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train a multilingual tokenizer")
    
    parser.add_argument(
        "--data_root", 
        type=str, 
        default="data/processed",
        help="Root directory with training data subdirectories"
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default="data/tokenizer",
        help="Directory to save the trained tokenizer"
    )
    parser.add_argument(
        "--vocab_size", 
        type=int, 
        default=32000,
        help="Vocabulary size for the tokenizer"
    )
    parser.add_argument(
        "--max_samples_per_dataset", 
        type=int, 
        default=100000,
        help="Maximum samples to use from each dataset"
    )
    parser.add_argument(
        "--samples_per_type", 
        type=str, 
        default="english=0.4,swedish=0.4,code=0.2",
        help="Sampling ratio for each dataset type, in format 'type=weight,type=weight'"
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

def load_jsonl(file_path):
    """Load data from a JSONL file."""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError:
                logging.warning(f"Could not parse line in {file_path}: {line}")
    return data

def sample_data(data, max_samples):
    """Sample data up to max_samples."""
    if len(data) <= max_samples:
        return data
    return random.sample(data, max_samples)

def parse_sampling_ratios(ratios_str):
    """Parse the sampling ratios string into a dictionary."""
    ratios = {}
    for part in ratios_str.split(','):
        key, value = part.split('=')
        ratios[key.strip()] = float(value.strip())
    
    # Normalize ratios
    total = sum(ratios.values())
    for key in ratios:
        ratios[key] /= total
    
    return ratios

def collect_training_data(data_root, max_samples_per_dataset, sampling_ratios, logger):
    """
    Collect training data from multiple datasets for tokenizer training.
    
    Args:
        data_root: Root directory containing subdirectories for each dataset type
        max_samples_per_dataset: Maximum number of samples to use from each dataset
        sampling_ratios: Dictionary of sampling ratios for each dataset type
        logger: Logger instance
        
    Returns:
        List of texts to train the tokenizer on
    """
    texts = []
    dataset_types = ["english", "swedish", "code"]
    samples_per_type = {}
    
    # Calculate samples per type based on ratios and total max samples
    total_samples = max_samples_per_dataset * len(dataset_types)
    for dtype in dataset_types:
        if dtype in sampling_ratios:
            samples_per_type[dtype] = int(total_samples * sampling_ratios[dtype])
        else:
            samples_per_type[dtype] = int(total_samples / len(dataset_types))
    
    # Collect texts from each dataset type
    for dataset_type in dataset_types:
        dataset_dir = os.path.join(data_root, dataset_type)
        if not os.path.exists(dataset_dir):
            logger.warning(f"Dataset directory {dataset_dir} does not exist. Skipping.")
            continue
            
        train_file = os.path.join(dataset_dir, "train.jsonl")
        if not os.path.exists(train_file):
            logger.warning(f"Training file {train_file} does not exist. Skipping.")
            continue
            
        logger.info(f"Loading data from {train_file}...")
        samples = load_jsonl(train_file)
        
        # Sample data based on the configured ratio
        max_for_this_type = samples_per_type.get(dataset_type, max_samples_per_dataset)
        sampled_data = sample_data(samples, max_for_this_type)
        
        logger.info(f"Selected {len(sampled_data)} samples from {dataset_type} dataset")
        
        # Extract texts from samples
        for sample in sampled_data:
            if 'text' in sample:
                texts.append(sample['text'])
            elif 'content' in sample:
                texts.append(sample['content'])
            else:
                logger.warning(f"Sample in {train_file} does not contain 'text' or 'content' field.")
    
    logger.info(f"Collected {len(texts)} texts for tokenizer training")
    return texts

def train_tokenizer(texts, vocab_size, output_dir, logger):
    """
    Train a tokenizer on the provided texts.
    
    Args:
        texts: List of texts to train on
        vocab_size: Size of the vocabulary
        output_dir: Directory to save the tokenizer
        logger: Logger instance
    """
    from tokenizers import ByteLevelBPETokenizer
    from transformers import PreTrainedTokenizerFast
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize a tokenizer
    tokenizer = ByteLevelBPETokenizer()
    
    # Train the tokenizer
    logger.info(f"Training tokenizer with vocabulary size {vocab_size}...")
    tokenizer.train_from_iterator(
        texts,
        vocab_size=vocab_size,
        min_frequency=2,
        special_tokens=["<s>", "</s>", "<unk>", "<pad>", "<mask>"]
    )
    
    # Save the tokenizer files
    tokenizer_path = os.path.join(output_dir, "tokenizer")
    tokenizer.save_model(output_dir, "tokenizer")
    
    logger.info(f"Tokenizer saved to {output_dir}")
    
    # Convert to transformers tokenizer and save
    pretrained_tokenizer = PreTrainedTokenizerFast(
        tokenizer_file=os.path.join(output_dir, "tokenizer.json"),
        bos_token="<s>",
        eos_token="</s>",
        unk_token="<unk>",
        pad_token="<pad>",
        mask_token="<mask>"
    )
    
    pretrained_tokenizer.save_pretrained(output_dir)
    logger.info(f"Converted tokenizer to transformers format and saved to {output_dir}")
    
    # Test the tokenizer
    test_sentence = "This is a test sentence to check our multilingual tokenizer."
    encoded = pretrained_tokenizer.encode(test_sentence)
    decoded = pretrained_tokenizer.decode(encoded)
    
    logger.info(f"Tokenizer test:")
    logger.info(f"Original: {test_sentence}")
    logger.info(f"Encoded: {encoded}")
    logger.info(f"Decoded: {decoded}")
    
    return pretrained_tokenizer

def main():
    """Main function to train a multilingual tokenizer."""
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
        # Parse sampling ratios
        sampling_ratios = parse_sampling_ratios(args.samples_per_type)
        logger.info(f"Using sampling ratios: {sampling_ratios}")
        
        # Collect training data
        logger.info("Collecting training data...")
        texts = collect_training_data(
            args.data_root,
            args.max_samples_per_dataset,
            sampling_ratios,
            logger
        )
        
        # Train tokenizer
        logger.info("Training tokenizer...")
        tokenizer = train_tokenizer(
            texts,
            args.vocab_size,
            args.output_dir,
            logger
        )
        
        logger.info("Tokenizer training completed successfully!")
        
    except Exception as e:
        logger.error(f"Error in tokenizer training: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main() 