#!/usr/bin/env python
"""
Download Swedish data directly from FineWeb-2 dataset using the Hugging Face API.
This approach bypasses datatrove and focuses on robust direct download.
"""

import os
import time
import logging
import tempfile
import shutil
import random
import pandas as pd
from tqdm import tqdm
from huggingface_hub import hf_hub_download, HfApi, configure_http_backend
import requests
from urllib3.util import Retry
from requests.adapters import HTTPAdapter

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# FineWeb-2 Swedish dataset information
REPO_ID = "HuggingFaceFW/fineweb-2"
REPO_TYPE = "dataset"
DATA_PATH_PREFIX = "data/swe_Latn/train"

def configure_http_session():
    """
    Configure HTTP session with robust retry strategy and timeouts
    for corporate networks with potential restrictions.
    """
    # Set environment variables for HuggingFace Hub
    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"  # Enable optimized transfer
    os.environ["HF_HUB_TIMEOUT"] = "300"           # 5 minute timeout
    
    # Update HF HTTP defaults using the official configure function
    configure_http_backend(
        timeout=30,
        retries=10,
        retry_factor=2,
        pool_size=10
    )
    
    # Configure requests for other HTTP operations
    retry_strategy = Retry(
        total=10,                      # Total number of retries
        backoff_factor=2,              # Exponential backoff
        status_forcelist=[408, 429, 500, 502, 503, 504],  # Retry on these status codes
        allowed_methods=["HEAD", "GET", "OPTIONS"]         # Only retry for these methods
    )
    
    # Create a session with the retry strategy
    session = requests.Session()
    adapter = HTTPAdapter(max_retries=retry_strategy, pool_connections=10, pool_maxsize=10)
    session.mount('http://', adapter)
    session.mount('https://', adapter)
    
    return session

def list_fineweb_files():
    """
    List all Swedish parquet files from the FineWeb-2 dataset.
    
    Returns:
        List of file paths in the dataset
    """
    logger.info(f"Listing files in {REPO_ID}/{DATA_PATH_PREFIX}")
    
    try:
        # Configure the HTTP backend first
        configure_http_session()
        
        # Create API client (API now uses global HTTP settings)
        api = HfApi()
        
        # Get list of files with pattern matching
        files = api.list_repo_files(
            repo_id=REPO_ID,
            repo_type=REPO_TYPE,
            patterns=[f"{DATA_PATH_PREFIX}/*.parquet"]
        )
        
        logger.info(f"Found {len(files)} Swedish parquet files in FineWeb-2")
        return files
        
    except Exception as e:
        logger.error(f"Error listing files: {e}")
        return []

def download_parquet_file(file_path, local_dir, retry_count=3):
    """
    Download a single parquet file with robust error handling.
    
    Args:
        file_path: Path to file in Hugging Face repository
        local_dir: Directory to save the file
        retry_count: Number of retries
    
    Returns:
        Path to downloaded file or None if failed
    """
    for attempt in range(retry_count):
        try:
            logger.info(f"Downloading {file_path} (attempt {attempt+1}/{retry_count})")
            
            # Set a longer timeout for this specific download
            os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "180"  # 3 minutes
            
            # Try to download with optimized settings
            local_file = hf_hub_download(
                repo_id=REPO_ID,
                filename=file_path,
                repo_type=REPO_TYPE,
                local_dir=local_dir,
                local_dir_use_symlinks=False,
                token=os.getenv("HF_TOKEN", None),  # Use token if available
                force_download=True  # Force download even if file exists
            )
            
            # Verify file exists and is not empty
            if os.path.exists(local_file) and os.path.getsize(local_file) > 0:
                logger.info(f"Successfully downloaded {file_path} to {local_file}")
                return local_file
            else:
                logger.warning(f"Downloaded file is empty or doesn't exist")
                continue
                
        except Exception as e:
            logger.warning(f"Error downloading {file_path}: {e}")
            if attempt < retry_count - 1:
                wait_time = (attempt + 1) * 5  # Increasing backoff
                logger.info(f"Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
            else:
                logger.error(f"Failed to download {file_path} after {retry_count} attempts")
                return None
    
    return None

def extract_text_from_parquet(parquet_file):
    """
    Extract text content from a parquet file.
    
    Args:
        parquet_file: Path to parquet file
    
    Returns:
        List of text documents
    """
    try:
        # Read the parquet file
        df = pd.read_parquet(parquet_file)
        
        # Extract the text column
        if 'text' in df.columns:
            return df['text'].tolist()
        else:
            logger.error(f"No 'text' column found in {parquet_file}")
            return []
            
    except Exception as e:
        logger.error(f"Error reading parquet file {parquet_file}: {e}")
        return []

def try_alternate_file_listing():
    """
    Alternative method to list files when the Hugging Face API fails.
    Uses a hardcoded list of known files in the repository.
    
    Returns:
        List of file paths in the dataset
    """
    logger.info("Using alternative file listing method")
    
    # Known file patterns in the FineWeb-2 dataset (using a predictable naming scheme)
    # Format: data/swe_Latn/train/000_00000.parquet, 000_00001.parquet, etc.
    known_files = []
    
    # Add first 15 files from shard 000 (these should exist in the repo)
    for i in range(15):
        file_path = f"{DATA_PATH_PREFIX}/000_{i:05d}.parquet"
        known_files.append(file_path)
    
    # Add first 5 files from other shards as fallback
    for shard in range(1, 10):
        for i in range(5):
            file_path = f"{DATA_PATH_PREFIX}/{shard:03d}_{i:05d}.parquet"
            known_files.append(file_path)
    
    logger.info(f"Using {len(known_files)} predefined file paths")
    return known_files

def download_swedish_data(output_dir, size_limit_gb, max_files=None):
    """
    Download Swedish data from FineWeb-2 dataset directly using Hugging Face API.
    
    Args:
        output_dir: Directory to save the data
        size_limit_gb: Maximum size in GB
        max_files: Maximum number of files to process (for testing)
    
    Returns:
        List of output files
    """
    logger.info(f"Downloading Swedish data from FineWeb-2 to {output_dir}")
    logger.info(f"Size limit: {size_limit_gb} GB")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Convert size limit to bytes
    size_limit_bytes = int(size_limit_gb * 1024**3)
    
    # Create temporary directory for downloads
    temp_dir = tempfile.mkdtemp(prefix="fineweb-swe-")
    logger.info(f"Using temporary directory: {temp_dir}")
    
    try:
        # List available files
        files = list_fineweb_files()
        
        # If file listing failed, try alternate method
        if not files:
            logger.warning("Standard file listing failed, trying alternate method")
            files = try_alternate_file_listing()
            
        if not files:
            logger.error("Could not get file list using any method")
            return []
            
        # Randomize order to get a diverse sample
        random.shuffle(files)
        
        # Limit number of files for testing if specified
        if max_files is not None:
            files = files[:max_files]
            
        # Output file
        output_file = os.path.join(output_dir, "swedish_data.txt")
        total_bytes = 0
        doc_count = 0
        
        # Process files until size limit is reached
        with open(output_file, "w", encoding="utf-8") as outfile:
            for file_path in tqdm(files, desc="Processing files"):
                # Check if we've already reached the size limit
                if total_bytes >= size_limit_bytes:
                    logger.info(f"Reached size limit of {size_limit_gb} GB")
                    break
                    
                # Download the file
                local_file = download_parquet_file(file_path, temp_dir)
                if not local_file:
                    logger.warning(f"Skipping {file_path} due to download error")
                    continue
                
                # Extract text from the file
                texts = extract_text_from_parquet(local_file)
                
                # Write documents to output file
                for text in texts:
                    # Skip empty texts
                    if not text or not isinstance(text, str):
                        continue
                        
                    # Get size of this document
                    doc_bytes = len(text.encode("utf-8"))
                    
                    # If adding this document would exceed the limit, stop
                    if total_bytes + doc_bytes > size_limit_bytes:
                        logger.info(f"Reached size limit of {size_limit_gb} GB")
                        break
                        
                    # Write document to file
                    outfile.write(text + "\n")
                    total_bytes += doc_bytes
                    doc_count += 1
                    
                    # Log progress periodically 
                    if doc_count % 1000 == 0:
                        logger.info(f"Processed {doc_count} documents, {total_bytes/1024**3:.2f} GB so far")
        
        # Check if we got any data
        if total_bytes == 0:
            logger.error("Failed to download any data")
            return []
            
        logger.info(f"Downloaded {doc_count} Swedish documents totaling {total_bytes/1024**3:.2f} GB to {output_file}")
        return [output_file]
        
    except Exception as e:
        logger.error(f"Error downloading Swedish data: {e}")
        return []
        
    finally:
        # Clean up temporary directory
        logger.info(f"Cleaning up temporary files in {temp_dir}")
        shutil.rmtree(temp_dir, ignore_errors=True)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Download Swedish data directly from FineWeb-2")
    parser.add_argument("--output_dir", type=str, default="data/raw/swe", help="Output directory")
    parser.add_argument("--size_limit", type=float, default=2.0, help="Size limit in GB")
    parser.add_argument("--max_files", type=int, default=None, help="Maximum number of files to process (for testing)")
    
    args = parser.parse_args()
    
    # Configure robust HTTP session
    configure_http_session()
    
    # Download the data
    download_swedish_data(args.output_dir, args.size_limit, args.max_files) 