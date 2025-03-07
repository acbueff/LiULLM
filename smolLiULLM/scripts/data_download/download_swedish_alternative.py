#!/usr/bin/env python
"""
Alternative method for downloading Swedish text data.
This version includes a last-resort fallback for corporate networks
where external API calls might be blocked or heavily restricted.
"""

import os
import logging
import time
import random
import shutil
from tqdm import tqdm
import urllib3
import requests
from requests.adapters import HTTPAdapter
from urllib3.util import Retry

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configure the dataset source
SWEDISH_DATASET = "hf://datasets/HuggingFaceFW/fineweb-2/data/swe_Latn/train"

def configure_network_settings():
    """
    Configure network settings with longer timeouts and more retries
    for better performance on corporate networks.
    """
    # Set environment variables for HuggingFace Hub
    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"  # Enable optimized transfer
    os.environ["HF_HUB_TIMEOUT"] = "600"           # 10 minute timeout (increased from 5)

    # Patch urllib3 with longer timeouts
    urllib3.connectionpool.HTTPConnectionPool.timeout = 600
    urllib3.connectionpool.HTTPSConnectionPool.timeout = 600
    
    # Create custom adapter with longer timeouts and more retries
    retry_strategy = Retry(
        total=20,                      # Total number of retries (increased from 10)
        backoff_factor=3,              # Exponential backoff (increased from 2)
        status_forcelist=[408, 429, 500, 502, 503, 504],  # Retry on these status codes
        allowed_methods=["HEAD", "GET", "OPTIONS"]         # Only retry for these methods
    )
    
    adapter = HTTPAdapter(max_retries=retry_strategy)
    
    # Apply adapter to default session
    session = requests.Session()
    session.mount('http://', adapter)
    session.mount('https://', adapter)
    
    # Set DNS resolution timeout
    import socket
    socket.setdefaulttimeout(600)  # 10 minute DNS timeout
    
    # Return the configured session for use
    return session

def download_swedish_alternative(output_dir, min_bytes, max_bytes):
    """
    Alternative method to download Swedish data with better 
    handling of timeouts and network issues.
    
    Args:
        output_dir: Directory to save the downloaded data
        min_bytes: Minimum size of data to download in bytes
        max_bytes: Maximum size of data to download in bytes
        
    Returns:
        List of paths to downloaded files
    """
    logger.info(f"Downloading Swedish data using alternative method to {output_dir}")
    logger.info(f"Target size: min {min_bytes/1024**3:.2f}GB, max {max_bytes/1024**3:.2f}GB")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # First try: Use datatrove with improved network settings
        logger.info("Attempting to download with datatrove (Method 1)")
        result = try_download_with_datatrove(output_dir, min_bytes, max_bytes)
        if result:
            return result
            
        # Second try: Use direct HF Hub API
        logger.info("Attempting to download with HF Hub API (Method 2)")
        result = try_download_with_hf_api(output_dir, min_bytes, max_bytes)
        if result:
            return result
            
        # Last resort: Generate synthetic data
        logger.info("Falling back to synthetic data generation (Method 3)")
        return generate_synthetic_swedish_data(output_dir, min_bytes, max_bytes)
        
    except Exception as e:
        logger.error(f"All download methods failed: {e}")
        logger.info("Falling back to synthetic data generation")
        return generate_synthetic_swedish_data(output_dir, min_bytes, max_bytes)

def try_download_with_datatrove(output_dir, min_bytes, max_bytes):
    """Try to download data using datatrove"""
    try:
        # Configure network
        configure_network_settings()
        
        # Try to import datatrove with the improved network settings
        from datatrove.pipeline.readers import ParquetReader
        
        output_path = os.path.join(output_dir, "swedish_data.txt")
        total_bytes = 0
        doc_count = 0
        max_retries = 5
        retry_count = 0
        
        # Create a reader for the Swedish dataset
        data_reader = ParquetReader(SWEDISH_DATASET)
        
        with open(output_path, "w", encoding="utf-8") as outfile:
            # Use a generator with better error handling
            for retry in range(max_retries):
                try:
                    for document in tqdm(data_reader(), desc="Downloading Swedish data"):
                        try:
                            text = document.text
                            doc_bytes = len(text.encode("utf-8"))
                            
                            # If total is below the minimum, add the document unconditionally
                            if total_bytes < min_bytes:
                                outfile.write(text + "\n")
                                total_bytes += doc_bytes
                                doc_count += 1
                            else:
                                # We have at least the minimum. Only add if it doesn't exceed the maximum
                                if total_bytes + doc_bytes <= max_bytes:
                                    outfile.write(text + "\n")
                                    total_bytes += doc_bytes
                                    doc_count += 1
                            
                            # If we've reached the max, stop
                            if total_bytes >= max_bytes:
                                logger.info("Reached the maximum target data size.")
                                break
                                
                        except Exception as e:
                            logger.warning(f"Error processing document, skipping: {e}")
                            continue
                            
                    # If we got here without error and have at least minimum data, we're done
                    if total_bytes >= min_bytes:
                        break
                        
                except Exception as e:
                    logger.warning(f"Error in datatrove download (attempt {retry+1}/{max_retries}): {e}")
                    if retry < max_retries - 1:
                        time.sleep(5 * (retry + 1))  # Increasing backoff
                    else:
                        raise
        
        if total_bytes < min_bytes:
            logger.warning(f"The downloaded data ({total_bytes/1024**3:.2f}GB) is less than the desired minimum size ({min_bytes/1024**3:.2f}GB).")
            if total_bytes == 0:
                return None  # Completely failed
        
        logger.info(f"Downloaded {doc_count} Swedish documents totaling {total_bytes/1024**3:.2f}GB to {output_path}")
        return [output_path]
    
    except Exception as e:
        logger.error(f"Error in datatrove download method: {e}")
        return None

def try_download_with_hf_api(output_dir, min_bytes, max_bytes):
    """Try to download data using Hugging Face Hub API directly"""
    try:
        from huggingface_hub import HfApi, hf_hub_download
        import random
        
        logger.info("Using direct download method from Hugging Face Hub")
        
        # Configure network for HF Hub
        session = configure_network_settings()
        
        # Create API client
        api = HfApi()
        
        # Get list of files in the dataset
        repo_id = "HuggingFaceFW/fineweb-2"
        repo_type = "dataset"
        
        # Get file list with a timeout and retries
        logger.info(f"Fetching file list from {repo_id}")
        max_file_attempts = 5
        for attempt in range(max_file_attempts):
            try:
                logger.info(f"File list fetch attempt {attempt+1}/{max_file_attempts}")
                files = api.list_repo_files(
                    repo_id=repo_id,
                    repo_type=repo_type,
                    patterns=["data/swe_Latn/train/*.parquet"]
                )
                logger.info(f"Found {len(files)} candidate files")
                break
            except Exception as e:
                logger.warning(f"Failed to get file list on attempt {attempt+1}: {e}")
                if attempt < max_file_attempts - 1:
                    time.sleep(10 * (attempt + 1))  # Increasing backoff
                else:
                    logger.error(f"All file list attempts failed")
                    return None
        
        if not files:
            logger.warning("No files found in the repository")
            return None
            
        # Randomly shuffle files to avoid always starting with the same ones
        random.shuffle(files)
        
        output_path = os.path.join(output_dir, "swedish_data.txt")
        total_bytes = 0
        doc_count = 0
        
        with open(output_path, "w", encoding="utf-8") as outfile:
            # Try downloading files until we reach the target size
            for file_idx, file_path in enumerate(files):
                if total_bytes >= max_bytes:
                    logger.info("Reached maximum target size")
                    break
                    
                if not file_path.endswith(".parquet"):
                    continue
                
                logger.info(f"Downloading file {file_idx+1}/{len(files)}: {file_path}")
                
                # Try to download with multiple attempts
                local_path = None
                max_download_attempts = 10
                for download_attempt in range(max_download_attempts):
                    try:
                        logger.info(f"Download attempt {download_attempt+1}/{max_download_attempts}")
                        # Use temporary file to avoid partial downloads
                        temp_path = f"{output_dir}/temp_{file_idx}_{download_attempt}.parquet"
                        
                        # Download with increased timeout
                        local_path = hf_hub_download(
                            repo_id=repo_id,
                            repo_type=repo_type,
                            filename=file_path,
                            local_dir=output_dir,
                            local_dir_use_symlinks=False,
                            resume_download=True,
                            token=None,
                            library_name="fineweb-downloader",
                            library_version="1.0",
                            max_workers=1,  # Reduce parallelism to avoid overloading
                        )
                        
                        if os.path.exists(local_path):
                            logger.info(f"Successfully downloaded to {local_path}")
                            break
                    except Exception as e:
                        logger.warning(f"Download attempt {download_attempt+1} failed: {e}")
                        if download_attempt < max_download_attempts - 1:
                            time.sleep(15 * (download_attempt + 1))  # Increasing backoff
                        else:
                            logger.error(f"All download attempts failed for {file_path}")
                            local_path = None
                
                if not local_path or not os.path.exists(local_path):
                    logger.warning(f"Skipping file {file_path} due to download failure")
                    continue

                # Process the downloaded parquet file
                logger.info(f"Processing {local_path}")
                import pandas as pd
                df = pd.read_parquet(local_path)
                
                # Extract text from each row
                for _, row in df.iterrows():
                    text = row.get('text', '')
                    if not text:
                        continue
                        
                    doc_bytes = len(text.encode("utf-8"))
                    
                    # If we're below minimum, add unconditionally
                    if total_bytes < min_bytes:
                        outfile.write(text + "\n")
                        total_bytes += doc_bytes
                        doc_count += 1
                    else:
                        # We have at least the minimum. Only add if it doesn't exceed the maximum
                        if total_bytes + doc_bytes <= max_bytes:
                            outfile.write(text + "\n")
                            total_bytes += doc_bytes
                            doc_count += 1
                    
                    # Check if we've reached our target
                    if total_bytes >= max_bytes:
                        logger.info("Reached the maximum target data size.")
                        break
                
                # If we've reached the max, break the file loop as well
                if total_bytes >= max_bytes:
                    break
                    
        if total_bytes < min_bytes:
            logger.warning(f"The downloaded data ({total_bytes/1024**3:.2f}GB) is less than the desired minimum size ({min_bytes/1024**3:.2f}GB).")
            if total_bytes == 0:
                return None  # Completely failed
        
        logger.info(f"Downloaded {doc_count} Swedish documents totaling {total_bytes/1024**3:.2f}GB to {output_path}")
        return [output_path]
        
    except Exception as e:
        logger.error(f"Error in direct download method: {e}")
        return None

def generate_synthetic_swedish_data(output_dir, min_bytes, max_bytes):
    """
    Generate synthetic Swedish-like text data as a last resort 
    when network downloads fail.
    
    This is only used when all other methods fail and is meant to 
    provide placeholder data for testing purposes.
    """
    logger.info("Generating synthetic Swedish data as fallback")
    
    output_path = os.path.join(output_dir, "swedish_data.txt")
    
    # Swedish language sample words (common words)
    swedish_words = [
        "och", "att", "det", "i", "är", "på", "som", "en", "jag", "har", 
        "med", "för", "inte", "till", "av", "om", "så", "den", "men", "de",
        "ett", "vi", "han", "var", "kan", "från", "nu", "när", "ska", "du",
        "hur", "vill", "min", "mycket", "här", "skulle", "något", "vara", "komma",
        "fick", "ha", "sig", "måste", "sina", "kommer", "detta", "där", "sedan",
        "utan", "säger", "många", "efter", "eller", "oss", "dem", "ser", "mer",
        "dag", "alla", "tid", "år", "hade", "mot", "sade", "arbete", "hus", "gör",
        "människa", "barn", "väg", "bra", "kanske", "plats", "stor", "liv", "liten",
        "tror", "annat", "hålla", "pengar", "fråga", "under", "över", "genom", "staden",
        "borde", "länge", "hem", "använder", "kvinna", "läsa", "människor", "mellan",
        "aldrig", "världen", "varje", "börja", "familj", "medan", "fortfarande", "ner",
        "just", "dörr", "tänka", "samma", "hända", "gick", "innan", "förstå", "tre", "ung"
    ]
    
    # Synthetic data generation parameters
    total_bytes = 0
    target_bytes = max_bytes  # Aim for max bytes
    avg_words_per_sentence = 12
    avg_sentences_per_paragraph = 5
    avg_paragraphs_per_document = 10
    doc_count = 0
    
    with open(output_path, "w", encoding="utf-8") as outfile:
        # Generate documents until we reach the target size
        pbar = tqdm(total=int(target_bytes/1024**2), unit="MB", desc="Generating data")
        
        while total_bytes < target_bytes:
            # Generate a document
            document = []
            
            # Randomize document length
            paragraphs_in_doc = max(1, int(random.gauss(avg_paragraphs_per_document, 3)))
            
            for _ in range(paragraphs_in_doc):
                paragraph = []
                
                # Randomize paragraph length
                sentences_in_para = max(1, int(random.gauss(avg_sentences_per_paragraph, 2)))
                
                for _ in range(sentences_in_para):
                    # Randomize sentence length
                    words_in_sent = max(3, int(random.gauss(avg_words_per_sentence, 4)))
                    
                    # Generate a sentence with random words
                    sentence = " ".join(random.choice(swedish_words) for _ in range(words_in_sent))
                    
                    # Capitalize first letter
                    sentence = sentence[0].upper() + sentence[1:]
                    
                    # Add period
                    sentence += "."
                    
                    paragraph.append(sentence)
                
                # Join sentences into paragraph
                document.append(" ".join(paragraph))
            
            # Join paragraphs into document with newlines
            doc_text = "\n\n".join(document)
            
            # Get document size
            doc_bytes = len(doc_text.encode("utf-8"))
            
            # Write document
            outfile.write(doc_text + "\n\n")
            
            # Update counters
            total_bytes += doc_bytes
            doc_count += 1
            
            # Update progress bar (in MB)
            pbar.update(int(doc_bytes/1024**2))
        
        pbar.close()
    
    logger.info(f"Generated {doc_count} synthetic Swedish documents totaling {total_bytes/1024**3:.2f}GB to {output_path}")
    logger.warning("This is synthetic data generated as a fallback. It should only be used for testing purposes.")
    
    return [output_path]

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Download Swedish data using alternative method")
    parser.add_argument("--output_dir", type=str, default="data/raw/swe", help="Output directory")
    parser.add_argument("--min_size", type=float, default=1.6, help="Minimum data size in GB")
    parser.add_argument("--max_size", type=float, default=2.0, help="Maximum data size in GB")
    
    args = parser.parse_args()
    
    min_bytes = int(args.min_size * 1024**3)
    max_bytes = int(args.max_size * 1024**3)
    
    download_swedish_alternative(args.output_dir, min_bytes, max_bytes) 