#!/usr/bin/env python3
"""
Data analysis script for determining optimal model size based on Chinchilla scaling laws.
This script analyzes multilingual datasets (English, Swedish, code) and recommends
appropriate model parameters based on total token counts.
"""

import os
import sys
import argparse
import logging
import json
import math
import yaml
from typing import Dict, Any, Tuple
from pathlib import Path
from tqdm import tqdm
from transformers import AutoTokenizer

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ""))

from src.utils.logging import setup_logging
from src.utils.config import load_config

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Analyze data and determine model scaling")
    
    parser.add_argument(
        "--data_root", 
        type=str, 
        default="data/processed",
        help="Root directory with training data subdirectories"
    )
    parser.add_argument(
        "--tokenizer_path", 
        type=str, 
        default="data/tokenizer",
        help="Path to tokenizer files"
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="configs/model_scaling.yaml",
        help="Output file to save scaling recommendations"
    )
    parser.add_argument(
        "--dataset_types",
        type=str,
        default="english,swedish,code",
        help="Comma-separated list of dataset types to analyze"
    )
    parser.add_argument(
        "--sample_limit",
        type=int,
        default=10000,
        help="Limit samples to count (for estimation with large datasets)"
    )
    parser.add_argument(
        "--token_estimation_factor",
        type=float,
        default=1.0,
        help="Factor to adjust token estimation if using a sample"
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

def load_jsonl(file_path, limit=None):
    """Load data from a JSONL file with optional limit."""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if limit and i >= limit:
                break
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError:
                logging.warning(f"Could not parse line in {file_path}: {line}")
    return data

def count_tokens_in_dataset(dataset_path, tokenizer, limit=None, logger=None):
    """
    Count tokens in a dataset using the tokenizer.
    
    Args:
        dataset_path: Path to the dataset file (JSONL format)
        tokenizer: Tokenizer to use for counting
        limit: Maximum number of samples to process (for estimation)
        logger: Logger instance
        
    Returns:
        Tuple of (total tokens, number of samples processed, average tokens per sample)
    """
    if not os.path.exists(dataset_path):
        if logger:
            logger.warning(f"Dataset file not found: {dataset_path}")
        return 0, 0, 0
    
    # Load the dataset
    samples = load_jsonl(dataset_path, limit)
    if not samples:
        if logger:
            logger.warning(f"No samples found in {dataset_path}")
        return 0, 0, 0
    
    total_tokens = 0
    sample_count = len(samples)
    
    for sample in tqdm(samples, desc=f"Counting tokens in {os.path.basename(dataset_path)}"):
        if 'text' in sample:
            text = sample['text']
        elif 'content' in sample:
            text = sample['content']
        else:
            if logger:
                logger.warning(f"Sample in {dataset_path} has no 'text' or 'content' field")
            continue
        
        # Count tokens
        tokens = tokenizer(text, return_tensors="pt")
        total_tokens += tokens.input_ids.size(1)
    
    avg_tokens = total_tokens / sample_count if sample_count > 0 else 0
    
    return total_tokens, sample_count, avg_tokens

def estimate_full_dataset_tokens(total_tokens, sample_count, total_file_count, estimation_factor=1.0):
    """
    Estimate the total tokens in the full dataset.
    
    Args:
        total_tokens: Total tokens in the sample
        sample_count: Number of samples processed
        total_file_count: Total number of lines in the file
        estimation_factor: Factor to adjust estimation
        
    Returns:
        Estimated total tokens in the full dataset
    """
    if sample_count == 0:
        return 0
    
    # Calculate tokens per sample
    tokens_per_sample = total_tokens / sample_count
    
    # Estimate total tokens
    estimated_total = tokens_per_sample * total_file_count * estimation_factor
    
    return int(estimated_total)

def count_file_lines(file_path):
    """Count the number of lines in a file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return sum(1 for _ in f)

def apply_chinchilla_scaling(total_tokens, logger=None):
    """
    Apply Chinchilla scaling law to determine optimal model size.
    
    Args:
        total_tokens: Total number of tokens across all datasets
        logger: Logger instance
        
    Returns:
        Dictionary with model scaling recommendations
    """
    # Chinchilla scaling law: compute optimal model size based on token count
    optimal_params = total_tokens / 20
    
    # Round to nearest power of 2 for practical implementation
    log2_params = math.log2(optimal_params)
    rounded_log2_params = round(log2_params)
    rounded_params = 2 ** rounded_log2_params
    
    # Calculate approximate model dimensions (assuming Transformer architecture)
    # For LLaMA-style models:
    # - num_params ≈ 12 * d_model^2 for a model with d_model dim and standard architecture
    approx_d_model = int(math.sqrt(rounded_params / 12))
    
    # Round to nearest multiple of 128 (typical for transformer models)
    d_model = round(approx_d_model / 128) * 128
    
    # Calculate other architecture parameters
    # These are approximate, based on common ratios in transformer models
    n_layers = max(int(d_model / 64), 8)  # Minimum 8 layers
    n_heads = max(d_model // 64, 8)  # Heads typically divide d_model by 64 or 128
    n_kv_heads = max(n_heads // 2, 1)  # KV heads are often fewer than attention heads
    intermediate_size = int(d_model * 2.6667)  # LLaMA uses 2.6667x for MLP size
    
    # Optimal training tokens according to Chinchilla is approximately 20x parameters
    optimal_training_tokens = rounded_params * 20
    
    # Create scaling recommendations
    scaling_recommendations = {
        'data_analysis': {
            'total_tokens_available': int(total_tokens),
            'optimal_parameter_count': int(rounded_params),
            'optimal_training_tokens': int(optimal_training_tokens)
        },
        'model_architecture': {
            'dim': int(d_model),
            'n_layers': int(n_layers),
            'n_heads': int(n_heads),
            'n_kv_heads': int(n_kv_heads),
            'intermediate_size': int(intermediate_size),
            'norm_eps': 1e-5,
        },
        'training_recommendations': {
            'batch_size': 512,  # This is a placeholder, would need hardware info
            'learning_rate': 3e-4,
            'weight_decay': 0.1,
            'epochs_to_train': math.ceil(optimal_training_tokens / total_tokens)
        }
    }
    
    if logger:
        logger.info(f"Chinchilla scaling law results:")
        logger.info(f"  Total available tokens: {total_tokens:,}")
        logger.info(f"  Optimal parameter count: {rounded_params:,}")
        logger.info(f"  Recommended model dimension: {d_model}")
        logger.info(f"  Recommended layers: {n_layers}")
        logger.info(f"  Recommended attention heads: {n_heads}")
        logger.info(f"  Optimal training tokens: {optimal_training_tokens:,}")
        logger.info(f"  Recommended epochs: {scaling_recommendations['training_recommendations']['epochs_to_train']}")
    
    return scaling_recommendations

def count_tokens_with_fallback(text, tokenizer):
    """Count tokens with fallback to character-based estimation if tokenizer fails."""
    try:
        tokens = tokenizer.encode(text)
        return len(tokens)
    except Exception as e:
        # Fallback to a simple character-based approximation
        return len(text.split()) + len(text) // 5  # Rough approximation

def main():
    """Main function to analyze data and determine scaling."""
    # Parse arguments
    args = parse_args()
    
    # Set up logging
    log_level = logging.DEBUG if args.debug else logging.INFO
    logger = setup_logging(
        log_dir=args.log_dir,
        log_level=log_level,
        experiment_name="data_analysis"
    )
    
    try:
        # Parse dataset types
        dataset_types = [t.strip() for t in args.dataset_types.split(',')]
        logger.info(f"Analyzing dataset types: {dataset_types}")
        
        # Handle multiple tokenizer loading approaches
        tokenizer = None
        tokenizer_path = os.path.abspath(args.tokenizer_path)
        logger.info(f"Looking for tokenizer at: {tokenizer_path}")
        
        # Try multiple approaches to load the tokenizer
        # 1. First, check if the path is directly to tokenizer.json
        if os.path.isfile(tokenizer_path) and tokenizer_path.endswith('.json'):
            try:
                logger.info(f"Loading tokenizer from file: {tokenizer_path}")
                from transformers import PreTrainedTokenizerFast
                tokenizer = PreTrainedTokenizerFast(tokenizer_file=tokenizer_path)
                logger.info("Successfully loaded tokenizer from file")
            except Exception as e:
                logger.error(f"Failed to load tokenizer from file: {e}")
        
        # 2. If that fails, check if it's a directory containing tokenizer.json
        if tokenizer is None and os.path.isdir(tokenizer_path):
            tokenizer_json = os.path.join(tokenizer_path, "tokenizer.json")
            if os.path.exists(tokenizer_json):
                try:
                    logger.info(f"Loading tokenizer from directory: {tokenizer_json}")
                    from transformers import PreTrainedTokenizerFast
                    tokenizer = PreTrainedTokenizerFast(tokenizer_file=tokenizer_json)
                    logger.info("Successfully loaded tokenizer from directory")
                except Exception as e:
                    logger.error(f"Failed to load tokenizer from directory: {e}")
        
        # 3. If all direct approaches fail, try to create a basic tokenizer as fallback
        if tokenizer is None:
            logger.warning("Could not load tokenizer. Using fallback GPT-2 tokenizer.")
            try:
                from transformers import GPT2Tokenizer
                tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
                logger.info("Using GPT-2 tokenizer as fallback")
            except Exception as e:
                logger.error(f"Failed to load fallback tokenizer: {e}")
                raise ValueError("Cannot proceed without a tokenizer")
        
        # Analyze each dataset type
        total_tokens_all_datasets = 0
        dataset_token_counts = {}
        
        for dataset_type in dataset_types:
            dataset_dir = os.path.join(args.data_root, dataset_type)
            
            if not os.path.exists(dataset_dir):
                logger.warning(f"Dataset directory not found: {dataset_dir}")
                continue
                
            # Check for training file
            train_file = os.path.join(dataset_dir, "train.jsonl")
            if not os.path.exists(train_file):
                logger.warning(f"Training file not found: {train_file}")
                continue
                
            logger.info(f"Analyzing {dataset_type} dataset...")
            
            # Count total lines in file for estimation
            total_lines = count_file_lines(train_file)
            logger.info(f"Total samples in {dataset_type}: {total_lines:,}")
            
            # Count tokens in sample
            sample_tokens, sample_count, avg_tokens = count_tokens_in_dataset(
                train_file, 
                tokenizer,
                limit=args.sample_limit if total_lines > args.sample_limit else None,
                logger=logger
            )
            
            # Estimate total tokens
            if sample_count < total_lines:
                estimated_total = estimate_full_dataset_tokens(
                    sample_tokens,
                    sample_count,
                    total_lines,
                    args.token_estimation_factor
                )
                logger.info(f"Sampled {sample_count:,} of {total_lines:,} samples")
                logger.info(f"Average tokens per sample: {avg_tokens:.2f}")
                logger.info(f"Estimated total tokens in {dataset_type}: {estimated_total:,}")
                dataset_token_counts[dataset_type] = estimated_total
                total_tokens_all_datasets += estimated_total
            else:
                logger.info(f"Total tokens in {dataset_type}: {sample_tokens:,}")
                logger.info(f"Average tokens per sample: {avg_tokens:.2f}")
                dataset_token_counts[dataset_type] = sample_tokens
                total_tokens_all_datasets += sample_tokens
        
        # Apply Chinchilla scaling
        logger.info(f"Total tokens across all datasets: {total_tokens_all_datasets:,}")
        scaling_recommendations = apply_chinchilla_scaling(total_tokens_all_datasets, logger)
        
        # Add dataset token counts to the output
        scaling_recommendations['data_analysis']['dataset_token_counts'] = dataset_token_counts
        
        # Save recommendations
        os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
        with open(args.output_file, 'w') as f:
            yaml.safe_dump(scaling_recommendations, f, sort_keys=False)
        
        logger.info(f"Scaling recommendations saved to {args.output_file}")
        
    except Exception as e:
        logger.error(f"Error in data analysis: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()