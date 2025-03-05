#!/usr/bin/env python3
"""
Local execution script for data preprocessing.
This script provides a simple way to preprocess training data locally.
"""

import os
import sys
import argparse
import logging
import time
import json
from pathlib import Path

# Add smolLiULLM directory to path
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, root_dir)

# Import smolLiULLM modules
from src.utils.logging import setup_logging
from src.utils.config import load_config
from src.data.preprocessing import TextPreprocessor

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Preprocess data for LLM training locally")
    
    parser.add_argument(
        "--config", 
        type=str, 
        default="configs/data_config.yaml",
        help="Path to data configuration file"
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        default="data/raw",
        help="Directory containing raw data files"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/processed",
        help="Directory to save processed data"
    )
    parser.add_argument(
        "--train_file",
        type=str,
        default="train.jsonl",
        help="Filename for processed training data"
    )
    parser.add_argument(
        "--val_file",
        type=str,
        default="validation.jsonl",
        help="Filename for processed validation data"
    )
    parser.add_argument(
        "--val_split",
        type=float,
        default=0.05,
        help="Validation split ratio"
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
    
    return parser.parse_args()

def setup_directories(directories):
    """Create directories if they don't exist."""
    for directory in directories:
        os.makedirs(directory, exist_ok=True)

def main():
    """Main function to run data preprocessing."""
    start_time = time.time()
    args = parse_args()
    
    # Setup logging
    log_level = logging.DEBUG if args.debug else logging.INFO
    log_dir = args.log_dir
    timestamp = int(start_time)
    setup_logging(log_dir=log_dir, log_level=log_level, experiment_name=f"preprocessing_{timestamp}")
    logger = logging.getLogger(__name__)
    
    # Create necessary directories
    setup_directories([args.log_dir, args.output_dir])
    
    logger.info("Starting data preprocessing")
    logger.info(f"Arguments: {args}")
    
    # Load configuration
    config = load_config(args.config)
    
    # Create data section if it doesn't exist
    if 'data' not in config:
        config['data'] = {}
    
    # Update config with CLI arguments if provided
    if args.input_dir:
        config['data']['input_dir'] = args.input_dir
    if args.output_dir:
        config['data']['output_dir'] = args.output_dir
    if args.train_file:
        config['data']['train_file'] = args.train_file
    if args.val_file:
        config['data']['val_file'] = args.val_file
    if args.val_split:
        config['data']['val_split'] = args.val_split
        
    # Update the raw_data section to use the provided input_dir
    if 'raw_data' not in config:
        config['raw_data'] = {}
    
    # Set paths for each language to the correct subdirectories
    if args.input_dir:
        config['raw_data']['english'] = os.path.join(args.input_dir, "eng")
        config['raw_data']['swedish'] = os.path.join(args.input_dir, "swe")
        config['raw_data']['code'] = os.path.join(args.input_dir, "code")
        config['raw_data']['icelandic'] = os.path.join(args.input_dir, "isl")
    
    # Update processed_data section with output paths
    if 'processed_data' not in config:
        config['processed_data'] = {}
    
    if args.output_dir and args.train_file:
        config['processed_data']['train'] = os.path.join(args.output_dir, args.train_file)
    if args.output_dir and args.val_file:
        config['processed_data']['validation'] = os.path.join(args.output_dir, args.val_file)
    
    # Update processing section with val_split if provided
    if 'processing' not in config:
        config['processing'] = {}
    
    if args.val_split:
        config['processing']['validation_split'] = args.val_split
    
    # Enhance preprocessing module with JSONL support
    # Patch the TextPreprocessor.process_language_data method to handle jsonl files
    original_process_directory = TextPreprocessor.process_directory
    
    def process_directory_with_jsonl(self, dir_path, file_pattern="*.txt"):
        """Add JSONL support to the process_directory method."""
        logger.info(f"Processing directory: {dir_path} with pattern {file_pattern}")
        all_lines = []
        
        if '.jsonl' in file_pattern:
            files = glob.glob(os.path.join(dir_path, file_pattern))
            logger.info(f"Found {len(files)} JSONL files in directory")
            
            for file_path in files:
                logger.info(f"Processing JSONL file: {file_path}")
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        for line in f:
                            self.stats['total_lines'] += 1
                            try:
                                data = json.loads(line)
                                # Extract text from JSONL - adjust field based on your data
                                text = data.get('text', '') or data.get('code', '') or data.get('content', '')
                                if not text:
                                    continue
                                
                                # Normalize text
                                normalized = self.normalize_text(text)
                                
                                # Apply quality filters
                                if self.is_valid_text(normalized):
                                    all_lines.append(normalized)
                                    self.stats['kept_lines'] += 1
                            except json.JSONDecodeError:
                                logger.warning(f"Invalid JSON line in {file_path}")
                except Exception as e:
                    logger.error(f"Error processing JSONL file {file_path}: {e}")
            
            return all_lines
        else:
            # Use the original method for other file types
            return original_process_directory(self, dir_path, file_pattern)
    
    # Patch the method
    import glob  # Import glob here to make it available to the patched method
    TextPreprocessor.process_directory = process_directory_with_jsonl
    
    # Patch the process_language_data method to use file_types config
    original_process_language_data = TextPreprocessor.process_language_data
    
    def process_language_data_with_file_types(self, lang):
        """Modified to use file_types configuration for each language."""
        # Reset stats for this language
        self.stats = defaultdict(int)
        self.stats['language'] = lang
        
        # Get directory path for this language
        dir_path = self.config['raw_data'].get(lang, "")
        if not dir_path or not os.path.exists(dir_path):
            logger.warning(f"Directory for {lang} not found: {dir_path}")
            return []
        
        # Get the appropriate file pattern for this language from config
        if 'file_types' in self.config and lang in self.config['file_types']:
            file_pattern = self.config['file_types'][lang]
        else:
            # Default patterns if not specified
            file_pattern = {
                'english': "*.parquet",
                'swedish': "*.txt",
                'icelandic': "*.txt",
                'code': "*.jsonl"
            }.get(lang, "*.txt")
        
        logger.info(f"Using file pattern {file_pattern} for {lang}")
        
        # Process all files in the directory with appropriate file pattern
        lines = self.process_directory(dir_path, file_pattern=file_pattern)
        
        # Deduplicate text if enabled
        if self.remove_duplicates:
            lines = self.deduplicate_text(lines)
        
        # Log statistics
        logger.info(f"Processed {lang} data:")
        for key, value in self.stats.items():
            if key != 'language':
                logger.info(f"  {key}: {value}")
        
        return lines
    
    # Import the required collections module for defaultdict
    from collections import defaultdict
    
    # Patch the method
    TextPreprocessor.process_language_data = process_language_data_with_file_types
    
    # Log the configuration
    logger.info(f"Data preprocessing configuration: {config}")
    
    # Initialize text preprocessor
    preprocessor = TextPreprocessor(config)
    
    # Preprocess data
    logger.info("Starting data preprocessing...")
    
    # Process all languages
    all_texts = {}
    for lang in ["english", "swedish", "code"]:
        if os.path.exists(config['raw_data'].get(lang, "")):
            logger.info(f"Processing {lang} data...")
            all_texts[lang] = preprocessor.process_language_data(lang)
        else:
            logger.info(f"No data found for {lang}")
    
    # Create train/val split
    train_path = os.path.join(args.output_dir, args.train_file)
    val_path = os.path.join(args.output_dir, args.val_file)
    
    # Create separate directories for each data type
    for lang in all_texts.keys():
        lang_dir = os.path.join(args.output_dir, lang)
        os.makedirs(lang_dir, exist_ok=True)
        
        # Update the output paths
        lang_train_path = os.path.join(lang_dir, args.train_file)
        lang_val_path = os.path.join(lang_dir, args.val_file)
        
        # Create the train/val split for this language
        lang_texts = {lang: all_texts[lang]}
        preprocessor.create_train_val_split(
            lang_texts, 
            lang_train_path, 
            lang_val_path, 
            validation_split=args.val_split,
            create_separate_files=False
        )
        
        logger.info(f"Created train/val split for {lang} in {lang_dir}")
    
    # Also create a combined train/val split
    preprocessor.create_train_val_split(
        all_texts, 
        train_path, 
        val_path, 
        validation_split=args.val_split
    )
    
    # Get counts
    train_samples = sum(len(texts) for texts in all_texts.values()) * (1 - args.val_split)
    val_samples = sum(len(texts) for texts in all_texts.values()) * args.val_split
    
    elapsed_time = time.time() - start_time
    logger.info(f"Data preprocessing completed in {elapsed_time:.2f} seconds")
    logger.info(f"Created {train_samples:.0f} training samples and {val_samples:.0f} validation samples")
    logger.info(f"Training data saved to: {os.path.join(args.output_dir, args.train_file)}")
    logger.info(f"Validation data saved to: {os.path.join(args.output_dir, args.val_file)}")
    
    # Log the language-specific directories
    for lang in all_texts.keys():
        logger.info(f"{lang} data saved to: {os.path.join(args.output_dir, lang)}")

if __name__ == "__main__":
    main() 