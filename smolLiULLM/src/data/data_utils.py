"""
Utility functions for data processing.
"""
import json
import logging
from typing import List, Dict, Any, Optional

def load_jsonl(file_path: str, limit: Optional[int] = None) -> List[Dict[str, Any]]:
    """
    Load data from a jsonl file.
    
    Args:
        file_path: Path to the jsonl file
        limit: Maximum number of lines to read (None for all)
        
    Returns:
        List of dictionaries, each representing a JSON object
    """
    data = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if limit is not None and i >= limit:
                    break
                try:
                    data.append(json.loads(line))
                except json.JSONDecodeError:
                    logging.warning(f"Error decoding JSON on line {i+1} in {file_path}")
    except Exception as e:
        logging.error(f"Error loading file {file_path}: {e}")
    
    return data

def count_tokens(text: str, tokenizer) -> int:
    """
    Count the number of tokens in a text using the provided tokenizer.
    
    Args:
        text: The text to tokenize
        tokenizer: The tokenizer to use
        
    Returns:
        Number of tokens
    """
    try:
        # Handle different tokenizer interfaces
        if hasattr(tokenizer, 'encode'):
            tokens = tokenizer.encode(text)
        elif hasattr(tokenizer, 'tokenize'):
            tokens = tokenizer.tokenize(text)
        else:
            # Fallback for other tokenizer interfaces
            tokens = tokenizer(text)
            
        # Handle different return types
        if isinstance(tokens, list):
            return len(tokens)
        elif hasattr(tokens, 'input_ids'):
            return len(tokens.input_ids[0])
        else:
            return len(tokens)
    except Exception as e:
        logging.warning(f"Error counting tokens: {e}")
        return 0 