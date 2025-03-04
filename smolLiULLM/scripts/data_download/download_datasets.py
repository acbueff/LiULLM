#!/usr/bin/env python3
"""
Dataset download script for the LiULLM pipeline.
Downloads data from HuggingFace repositories using direct URLs.
"""

import os
import argparse
import logging
import requests
import sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from datatrove.pipeline.readers import ParquetReader
from huggingface_hub import HfApi
from tqdm import tqdm

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), ""))

from src.utils.logging import setup_logging

# Constants
ENGLISH_REPO = "PleIAs/English-PD"
SWEDISH_DATASET = "hf://datasets/HuggingFaceFW/fineweb-2/data/swe_Latn/train"
ICELANDIC_REPO = "PleIAs/Icelandic-PD"  # Example, replace with actual repo

# Data size limits in bytes
DEFAULT_SIZE_LIMITS = {
    "english": 5 * 1024**3,  # 5GB
    "swedish": 3 * 1024**3,  # 3GB
    "icelandic": 2 * 1024**3,  # 2GB
}

logger = logging.getLogger(__name__)

def get_file_size(repo, filename):
    """
    Retrieve the file size (in bytes) using a GET request with a Range header.
    """
    url = f"https://huggingface.co/datasets/{repo}/resolve/main/{filename}"
    headers = {"Range": "bytes=0-0"}
    
    try:
        response = requests.get(url, headers=headers, stream=True)
        if response.status_code in (200, 206):  # 206 means partial content
            # Try to extract size from Content-Range header
            content_range = response.headers.get("Content-Range")
            if content_range:
                try:
                    # "Content-Range" is typically "bytes 0-0/12345"
                    total_size = int(content_range.split('/')[-1])
                    return total_size
                except (IndexError, ValueError):
                    pass
            # Fallback to the Content-Length header if available
            return int(response.headers.get("Content-Length", 0))
        else:
            logger.warning(f"Unable to fetch size for {filename}. Status code: {response.status_code}")
            return 0
    except Exception as e:
        logger.error(f"Error fetching size for {filename}: {e}")
        return 0

def download_file(repo, filename, output_dir):
    """
    Download a single file from a HuggingFace dataset repository.
    """
    url = f"https://huggingface.co/datasets/{repo}/resolve/main/{filename}"
    local_path = os.path.join(output_dir, filename)
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    
    try:
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            total_size = int(r.headers.get("Content-Length", 0))
            
            with open(local_path, "wb") as f:
                with tqdm(total=total_size, unit="B", unit_scale=True, desc=filename) as pbar:
                    for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk)
                        pbar.update(len(chunk))
        
        logger.info(f"Downloaded {filename} to {local_path}")
        return local_path
    except Exception as e:
        logger.error(f"Error downloading {filename}: {e}")
        return None

def download_english_data(output_dir, size_limit):
    """
    Download English data from PleIAs/English-PD repository.
    """
    logger.info(f"Downloading English data to {output_dir} (limit: {size_limit/1024**3:.2f}GB)")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        api = HfApi()
        # List all files in the repository
        files = api.list_repo_files(ENGLISH_REPO, repo_type="dataset")
        logger.info(f"Found {len(files)} files in {ENGLISH_REPO}")
        
        # Get file sizes
        file_sizes = {}
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = {executor.submit(get_file_size, ENGLISH_REPO, f): f for f in files}
            for future in tqdm(futures, desc="Getting file sizes"):
                filename = futures[future]
                size = future.result()
                file_sizes[filename] = size
        
        # Select files until size limit is reached
        selected_files = []
        total_size = 0
        for f in files:
            size = file_sizes.get(f, 0)
            if total_size + size <= size_limit:
                selected_files.append(f)
                total_size += size
            else:
                # If we're not close to the limit yet, keep trying more files
                if total_size < 0.9 * size_limit:
                    continue
                break
        
        logger.info(f"Selected {len(selected_files)} files for download (total: {total_size/1024**3:.2f}GB)")
        
        # Download files
        downloaded_files = []
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = {executor.submit(download_file, ENGLISH_REPO, f, output_dir): f for f in selected_files}
            for future in tqdm(futures, desc="Downloading files"):
                result = future.result()
                if result:
                    downloaded_files.append(result)
        
        logger.info(f"Successfully downloaded {len(downloaded_files)} English files")
        return downloaded_files
    
    except Exception as e:
        logger.error(f"Error downloading English data: {e}")
        return []

def download_swedish_data(output_dir, min_bytes, max_bytes):
    """
    Download Swedish data from FineWeb2 dataset using datatrove.
    """
    logger.info(f"Downloading Swedish data to {output_dir} (min: {min_bytes/1024**3:.2f}GB, max: {max_bytes/1024**3:.2f}GB)")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    output_path = os.path.join(output_dir, "swedish_data.txt")
    total_bytes = 0
    doc_count = 0
    
    try:
        # Create a reader for the Swedish dataset
        data_reader = ParquetReader(SWEDISH_DATASET)
        
        with open(output_path, "w", encoding="utf-8") as outfile:
            for document in tqdm(data_reader(), desc="Downloading Swedish data"):
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
        
        if total_bytes < min_bytes:
            logger.warning(f"The downloaded data ({total_bytes/1024**3:.2f}GB) is less than the desired minimum size ({min_bytes/1024**3:.2f}GB).")
        
        logger.info(f"Downloaded {doc_count} Swedish documents totaling {total_bytes/1024**3:.2f}GB to {output_path}")
        return [output_path]
    
    except Exception as e:
        logger.error(f"Error downloading Swedish data: {e}")
        return []

def download_icelandic_data(output_dir, size_limit):
    """
    Download Icelandic data.
    Note: This uses the same approach as the English data download.
    Replace ICELANDIC_REPO with the actual repository if different.
    """
    logger.info(f"Downloading Icelandic data to {output_dir} (limit: {size_limit/1024**3:.2f}GB)")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        api = HfApi()
        # List all files in the repository
        files = api.list_repo_files(ICELANDIC_REPO, repo_type="dataset")
        logger.info(f"Found {len(files)} files in {ICELANDIC_REPO}")
        
        # Get file sizes
        file_sizes = {}
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = {executor.submit(get_file_size, ICELANDIC_REPO, f): f for f in files}
            for future in tqdm(futures, desc="Getting file sizes"):
                filename = futures[future]
                size = future.result()
                file_sizes[filename] = size
        
        # Select files until size limit is reached
        selected_files = []
        total_size = 0
        for f in files:
            size = file_sizes.get(f, 0)
            if total_size + size <= size_limit:
                selected_files.append(f)
                total_size += size
            else:
                # If we're not close to the limit yet, keep trying more files
                if total_size < 0.9 * size_limit:
                    continue
                break
        
        logger.info(f"Selected {len(selected_files)} files for download (total: {total_size/1024**3:.2f}GB)")
        
        # Download files
        downloaded_files = []
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = {executor.submit(download_file, ICELANDIC_REPO, f, output_dir): f for f in selected_files}
            for future in tqdm(futures, desc="Downloading files"):
                result = future.result()
                if result:
                    downloaded_files.append(result)
        
        logger.info(f"Successfully downloaded {len(downloaded_files)} Icelandic files")
        return downloaded_files
    
    except Exception as e:
        logger.error(f"Error downloading Icelandic data: {e}")
        return []

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Download multilingual datasets from HuggingFace")
    
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/raw",
        help="Base directory to save downloaded data"
    )
    parser.add_argument(
        "--languages",
        nargs="+",
        choices=["english", "swedish", "icelandic", "all"],
        default=["all"],
        help="Languages to download"
    )
    parser.add_argument(
        "--english_size_gb",
        type=float,
        default=DEFAULT_SIZE_LIMITS["english"] / 1024**3,
        help="Maximum size of English data in GB"
    )
    parser.add_argument(
        "--swedish_min_gb",
        type=float,
        default=DEFAULT_SIZE_LIMITS["swedish"] * 0.8 / 1024**3,
        help="Minimum size of Swedish data in GB"
    )
    parser.add_argument(
        "--swedish_max_gb",
        type=float,
        default=DEFAULT_SIZE_LIMITS["swedish"] / 1024**3,
        help="Maximum size of Swedish data in GB"
    )
    parser.add_argument(
        "--icelandic_size_gb",
        type=float,
        default=DEFAULT_SIZE_LIMITS["icelandic"] / 1024**3,
        help="Maximum size of Icelandic data in GB"
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
    """Main function to download datasets."""
    # Parse arguments
    args = parse_args()
    
    # Set up logging
    log_level = logging.DEBUG if args.debug else logging.INFO
    logger = setup_logging(
        log_dir=args.log_dir,
        log_level=log_level,
        experiment_name="data_download"
    )
    
    # Determine which languages to download
    languages = args.languages
    if "all" in languages:
        languages = ["english", "swedish", "icelandic"]
    
    # Create base output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    all_results = {}
    
    # English
    if "english" in languages:
        english_output_dir = os.path.join(args.output_dir, "eng")
        english_size_bytes = int(args.english_size_gb * 1024**3)
        english_files = download_english_data(english_output_dir, english_size_bytes)
        all_results["english"] = english_files
    
    # Swedish
    if "swedish" in languages:
        swedish_output_dir = os.path.join(args.output_dir, "swe")
        swedish_min_bytes = int(args.swedish_min_gb * 1024**3)
        swedish_max_bytes = int(args.swedish_max_gb * 1024**3)
        swedish_files = download_swedish_data(swedish_output_dir, swedish_min_bytes, swedish_max_bytes)
        all_results["swedish"] = swedish_files
    
    # Icelandic
    if "icelandic" in languages:
        icelandic_output_dir = os.path.join(args.output_dir, "isl")
        icelandic_size_bytes = int(args.icelandic_size_gb * 1024**3)
        icelandic_files = download_icelandic_data(icelandic_output_dir, icelandic_size_bytes)
        all_results["icelandic"] = icelandic_files
    
    # Print summary
    logger.info("Download Summary:")
    for language, files in all_results.items():
        logger.info(f"{language.capitalize()}: Downloaded {len(files)} files")
    
    logger.info(f"All data downloaded to {args.output_dir}")

if __name__ == "__main__":
    main() 