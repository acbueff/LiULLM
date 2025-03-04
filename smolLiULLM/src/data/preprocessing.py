"""
Data preprocessing module for text cleaning and normalization.
"""

import os
import glob
import logging
import re
import ftfy
import hashlib
import random
from typing import List, Dict, Any, Tuple, Optional, Union
from collections import defaultdict
import unicodedata

logger = logging.getLogger(__name__)

class TextPreprocessor:
    """
    Text preprocessing class for cleaning and normalizing text data.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize text preprocessor with configuration.
        
        Args:
            config: Configuration dictionary with preprocessing options.
        """
        self.config = config
        self.remove_duplicates = config['processing'].get('remove_duplicates', True)
        self.lowercase = config['processing'].get('lowercase', False)
        self.min_length = config['processing'].get('min_length', 5)
        self.max_length = config['processing'].get('max_length', 2048)
        self.enable_length_filter = config['quality'].get('enable_length_filter', True)
        self.enable_char_filter = config['quality'].get('enable_char_filter', True)
        self.non_text_chars_ratio = config['quality'].get('non_text_chars_ratio', 0.3)
        
        # Compile regex patterns
        self.whitespace_regex = re.compile(r'\s+')
        self.non_text_regex = re.compile(r'[^a-zA-Z0-9áéíóúýðþæöåäëïüÿãõñçèàìòùş\s.,;:!?\'"-]')
        
        # Track statistics
        self.stats = defaultdict(int)
        
    def normalize_text(self, text: str) -> str:
        """
        Normalize text with unicode normalization, whitespace normalization, etc.
        
        Args:
            text: Input text to normalize.
            
        Returns:
            Normalized text.
        """
        # Skip empty text
        if not text or not text.strip():
            return ""
        
        # Fix encoding issues
        text = ftfy.fix_text(text)
        
        # Normalize unicode
        text = unicodedata.normalize('NFC', text)
        
        # Lowercase if configured
        if self.lowercase:
            text = text.lower()
        
        # Normalize whitespace
        text = self.whitespace_regex.sub(' ', text)
        
        # Strip leading/trailing whitespace
        text = text.strip()
        
        return text
    
    def is_valid_text(self, text: str) -> bool:
        """
        Check if the text passes quality filters.
        
        Args:
            text: Text to validate.
            
        Returns:
            True if the text passes all quality filters, False otherwise.
        """
        # Empty text
        if not text or not text.strip():
            self.stats['filtered_empty'] += 1
            return False
        
        # Length filter
        if self.enable_length_filter:
            # Split text into words
            words = text.split()
            if len(words) < self.min_length:
                self.stats['filtered_too_short'] += 1
                return False
                
        # Character filter
        if self.enable_char_filter:
            # Check for high proportion of non-text characters
            non_text_matches = self.non_text_regex.findall(text)
            if len(non_text_matches) / len(text) > self.non_text_chars_ratio:
                self.stats['filtered_non_text'] += 1
                return False
        
        return True
    
    def compute_hash(self, text: str) -> str:
        """
        Compute a hash for text deduplication.
        
        Args:
            text: Text to hash.
            
        Returns:
            Hash string.
        """
        return hashlib.md5(text.encode('utf-8')).hexdigest()
    
    def process_file(self, file_path: str) -> List[str]:
        """
        Process a single text file.
        
        Args:
            file_path: Path to the text file.
            
        Returns:
            List of processed text lines that pass filters.
        """
        logger.info(f"Processing file: {file_path}")
        processed_lines = []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    self.stats['total_lines'] += 1
                    
                    # Normalize text
                    normalized = self.normalize_text(line)
                    
                    # Apply quality filters
                    if self.is_valid_text(normalized):
                        processed_lines.append(normalized)
                        self.stats['kept_lines'] += 1
        except Exception as e:
            logger.error(f"Error processing file {file_path}: {e}")
            
        return processed_lines
    
    def process_directory(self, dir_path: str, file_pattern: str = "*.txt") -> List[str]:
        """
        Process all text files in a directory.
        
        Args:
            dir_path: Directory path containing text files.
            file_pattern: Glob pattern for text files.
            
        Returns:
            List of processed text lines from all files.
        """
        logger.info(f"Processing directory: {dir_path}")
        all_lines = []
        
        # Get list of files
        if '.parquet' in file_pattern:
            try:
                import pandas as pd
                files = glob.glob(os.path.join(dir_path, file_pattern))
                logger.info(f"Found {len(files)} parquet files in directory")
                
                # Process each parquet file
                for file_path in files:
                    logger.info(f"Processing parquet file: {file_path}")
                    try:
                        df = pd.read_parquet(file_path)
                        # Assuming the text is in a column named 'text' - adjust if needed
                        text_column = 'text' if 'text' in df.columns else df.columns[0]
                        for text in df[text_column]:
                            self.stats['total_lines'] += 1
                            # Normalize text
                            normalized = self.normalize_text(text)
                            # Apply quality filters
                            if self.is_valid_text(normalized):
                                all_lines.append(normalized)
                                self.stats['kept_lines'] += 1
                    except Exception as e:
                        logger.error(f"Error processing parquet file {file_path}: {e}")
            except ImportError:
                logger.error("Could not import pandas library which is required for parquet processing.")
                return []
        else:
            # Standard text file processing
            files = glob.glob(os.path.join(dir_path, file_pattern))
            logger.info(f"Found {len(files)} files in directory")
            
            # Process each file
            for file_path in files:
                processed_lines = self.process_file(file_path)
                all_lines.extend(processed_lines)
            
        return all_lines
    
    def deduplicate_text(self, lines: List[str]) -> List[str]:
        """
        Remove duplicate text lines.
        
        Args:
            lines: List of text lines.
            
        Returns:
            Deduplicated list of text lines.
        """
        if not self.remove_duplicates:
            return lines
            
        logger.info(f"Deduplicating {len(lines)} text lines")
        self.stats['before_dedup'] = len(lines)
        
        # Track seen hashes
        seen_hashes = set()
        deduplicated = []
        
        for line in lines:
            line_hash = self.compute_hash(line)
            if line_hash not in seen_hashes:
                seen_hashes.add(line_hash)
                deduplicated.append(line)
        
        self.stats['after_dedup'] = len(deduplicated)
        self.stats['duplicates_removed'] = self.stats['before_dedup'] - self.stats['after_dedup']
        
        logger.info(f"Removed {self.stats['duplicates_removed']} duplicate lines")
        return deduplicated
    
    def process_language_data(self, lang: str) -> List[str]:
        """
        Process text data for a specific language.
        
        Args:
            lang: Language identifier (e.g., 'icelandic', 'swedish', 'english')
            
        Returns:
            List of processed text lines for the language.
        """
        # Reset stats for this language
        self.stats = defaultdict(int)
        self.stats['language'] = lang
        
        # Get directory path for this language
        dir_path = self.config['raw_data'].get(lang, "")
        if not dir_path or not os.path.exists(dir_path):
            logger.warning(f"Directory for {lang} not found: {dir_path}")
            return []
        
        # Process all files in the directory with appropriate file pattern
        file_pattern = "*.parquet" if lang == "english" else "*.txt"
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

    def create_train_val_split(
        self, 
        all_texts: Dict[str, List[str]],
        output_train_path: str,
        output_val_path: str,
        validation_split: float = 0.02,
        create_separate_files: bool = True
    ) -> None:
        """
        Create training and validation splits from processed texts.
        
        Args:
            all_texts: Dictionary of language to list of processed texts.
            output_train_path: Path to save training data.
            output_val_path: Path to save validation data.
            validation_split: Fraction of data to use for validation.
            create_separate_files: Whether to create separate files for each language.
        """
        logger.info("Creating train/validation split")
        
        # Create output directories if they don't exist
        os.makedirs(os.path.dirname(output_train_path), exist_ok=True)
        os.makedirs(os.path.dirname(output_val_path), exist_ok=True)
        
        # Prepare train and validation data for each language
        train_texts = []
        val_texts = []
        
        for lang, texts in all_texts.items():
            random.shuffle(texts)  # Shuffle texts for each language
            
            # Determine split point
            split_idx = max(1, int(len(texts) * validation_split))
            
            # Split the data
            lang_val = texts[:split_idx]
            lang_train = texts[split_idx:]
            
            logger.info(f"{lang}: {len(lang_train)} training, {len(lang_val)} validation")
            
            # Create separate files for each language if requested
            if create_separate_files:
                # Generate language-specific output paths
                lang_train_path = output_train_path.replace('.jsonl', f'_{lang}.jsonl')
                lang_val_path = output_val_path.replace('.jsonl', f'_{lang}.jsonl')
                
                # Write language-specific train and validation data
                with open(lang_train_path, 'w', encoding='utf-8') as f:
                    for line in lang_train:
                        f.write(line + '\n')
                
                with open(lang_val_path, 'w', encoding='utf-8') as f:
                    for line in lang_val:
                        f.write(line + '\n')
                
                logger.info(f"Wrote {len(lang_train)} lines to {lang_train_path}")
                logger.info(f"Wrote {len(lang_val)} lines to {lang_val_path}")
            
            # Add to the global train/val sets
            train_texts.extend(lang_train)
            val_texts.extend(lang_val)
        
        # Shuffle again before writing
        random.shuffle(train_texts)
        random.shuffle(val_texts)
        
        # Write combined train and validation data
        with open(output_train_path, 'w', encoding='utf-8') as f:
            for line in train_texts:
                f.write(line + '\n')
        
        with open(output_val_path, 'w', encoding='utf-8') as f:
            for line in val_texts:
                f.write(line + '\n')
        
        logger.info(f"Wrote {len(train_texts)} lines to {output_train_path}")
        logger.info(f"Wrote {len(val_texts)} lines to {output_val_path}")

    def process_all_languages(self) -> None:
        """
        Process all language data and create train/val split.
        """
        logger.info("Starting data preprocessing for all languages")
        
        all_texts = {}
        total_processed = 0
        
        # Process each language
        for lang in self.config['raw_data'].keys():
            # Apply configured sampling ratio
            sampling_ratio = self.config['sampling'].get(lang, 1.0)
            
            # Process language data
            texts = self.process_language_data(lang)
            
            # Apply sampling if ratio < 1.0
            if sampling_ratio < 1.0:
                num_to_keep = max(1, int(len(texts) * sampling_ratio))
                random.shuffle(texts)
                texts = texts[:num_to_keep]
                logger.info(f"Sampled {lang} data to {len(texts)} texts (ratio: {sampling_ratio})")
            
            all_texts[lang] = texts
            total_processed += len(texts)
        
        logger.info(f"Total processed texts across all languages: {total_processed}")
        
        # Create train/validation split
        output_train_path = self.config['processed_data']['train']
        output_val_path = self.config['processed_data']['validation']
        validation_split = self.config['processing'].get('validation_split', 0.02)
        create_separate_files = self.config['processing'].get('create_separate_files', True)
        
        self.create_train_val_split(
            all_texts, 
            output_train_path, 
            output_val_path, 
            validation_split,
            create_separate_files
        ) 