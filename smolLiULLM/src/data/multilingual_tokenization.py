"""
Multilingual tokenization utilities extending the base TokenizerTrainer class.
"""

import os
import logging
import json
import random
from typing import Dict, List, Any
from pathlib import Path
from tqdm import tqdm

from src.data.tokenization import TokenizerTrainer

logger = logging.getLogger(__name__)

def load_jsonl(file_path):
    """Load data from a JSONL file or plain text file."""
    data = []
    is_plain_text = False
    
    with open(file_path, 'r', encoding='utf-8') as f:
        # Try to determine file format from first line
        first_line = f.readline().strip()
        f.seek(0)  # Reset file pointer
        
        try:
            # Try parsing as JSON
            json.loads(first_line)
            # If successful, process as JSONL
            for line in f:
                try:
                    data.append(json.loads(line))
                except json.JSONDecodeError:
                    logger.warning(f"Could not parse line in {file_path} as JSON, treating as plain text")
                    is_plain_text = True
                    break
        except json.JSONDecodeError:
            # Not valid JSON, treat as plain text
            is_plain_text = True
    
    # If plain text detected, reload and process as plain text
    if is_plain_text:
        logger.info(f"Processing {file_path} as plain text file (not JSONL)")
        data = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():  # Skip empty lines
                    data.append({"text": line.strip()})
    
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

class MultilingualTokenizerTrainer(TokenizerTrainer):
    """Extended TokenizerTrainer with support for multilingual dataset balancing."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize with extended configuration for multilingual support.
        
        Args:
            config: Configuration dictionary with multilingual settings
        """
        super().__init__(config)
        
        # Additional multilingual configs
        self.data_root = config.get('data_root', 'data/processed')
        self.max_samples_per_dataset = config.get('max_samples_per_dataset', 100000)
        self.sampling_ratios = config.get('sampling_ratios', {
            'english': 0.4,
            'swedish': 0.4,
            'code': 0.2
        })
        self.dataset_types = config.get('dataset_types', ['english', 'swedish', 'code'])
        
    def collect_training_data(self):
        """
        Collect balanced training data from multiple language/domain datasets.
        
        Returns:
            List of texts to train the tokenizer on
        """
        texts = []
        samples_per_type = {}
        
        # Calculate samples per type based on ratios and total max samples
        total_samples = self.max_samples_per_dataset * len(self.dataset_types)
        for dtype in self.dataset_types:
            if dtype in self.sampling_ratios:
                samples_per_type[dtype] = int(total_samples * self.sampling_ratios[dtype])
            else:
                samples_per_type[dtype] = int(total_samples / len(self.dataset_types))
        
        # Collect texts from each dataset type
        for dataset_type in self.dataset_types:
            dataset_dir = os.path.join(self.data_root, dataset_type)
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
            max_for_this_type = samples_per_type.get(dataset_type, self.max_samples_per_dataset)
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
    
    def train_from_multilingual(self):
        """
        Train a tokenizer on multilingual data with balanced sampling.
        
        Returns:
            Trained tokenizer
        """
        logger.info("Collecting multilingual training data...")
        texts = self.collect_training_data()
        
        # Create a temporary file with the collected texts
        temp_file = os.path.join(self.output_dir, "multilingual_training_data.txt")
        with open(temp_file, 'w', encoding='utf-8') as f:
            for text in texts:
                f.write(text + "\n")
        
        logger.info(f"Training tokenizer on {len(texts)} multilingual texts...")
        tokenizer = self.train([temp_file])
        
        # Clean up temporary file
        if os.path.exists(temp_file):
            os.remove(temp_file)
        
        return tokenizer 