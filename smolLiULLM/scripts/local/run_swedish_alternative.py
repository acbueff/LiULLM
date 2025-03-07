#!/usr/bin/env python
"""
Wrapper script to download Swedish data using the alternative method.
This script solves timeout issues on corporate networks.
"""

import os
import sys
import argparse
import logging
import importlib.util

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_module_from_path(module_name, file_path):
    """
    Load a Python module from a file path.
    
    Args:
        module_name: Name to give the module
        file_path: Path to the Python file
        
    Returns:
        The loaded module object
    """
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Download Swedish data using alternative method")
    
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/raw/swe",
        help="Directory to save downloaded data"
    )
    parser.add_argument(
        "--size_limit",
        type=float,
        default=2.0,
        help="Size limit in GB for downloaded data"
    )
    
    return parser.parse_args()

def main():
    """Main function to run Swedish data collection using alternative method."""
    args = parse_args()
    
    # Import the alternative download function using direct file path
    try:
        # Get the path to the alternative download script
        script_dir = os.path.dirname(os.path.abspath(__file__))
        alt_script_path = os.path.abspath(os.path.join(
            script_dir, "..", "data_download", "download_swedish_alternative.py"
        ))
        
        if not os.path.exists(alt_script_path):
            logger.error(f"Alternative download script not found at: {alt_script_path}")
            return 1
            
        logger.info(f"Loading alternative download script from: {alt_script_path}")
        
        # Load the module from the file path
        alt_module = load_module_from_path("download_swedish_alternative", alt_script_path)
        download_swedish_alternative = alt_module.download_swedish_alternative
        
        # Calculate min and max bytes (80% of size_limit to size_limit)
        min_bytes = int(args.size_limit * 0.8 * 1024**3)
        max_bytes = int(args.size_limit * 1024**3)
        
        # Create output directory
        os.makedirs(args.output_dir, exist_ok=True)
        
        logger.info(f"Starting Swedish data collection using alternative method")
        logger.info(f"Output directory: {args.output_dir}")
        logger.info(f"Size limit: {args.size_limit} GB (min: {min_bytes/1024**3:.2f} GB, max: {max_bytes/1024**3:.2f} GB)")
        
        # Run the alternative download
        result_files = download_swedish_alternative(args.output_dir, min_bytes, max_bytes)
        
        if result_files:
            logger.info(f"Swedish data collection completed successfully. Files created: {result_files}")
            return 0
        else:
            logger.error("Swedish data collection failed to produce any output files.")
            return 1
            
    except ImportError as e:
        logger.error(f"Failed to import necessary modules: {e}")
        return 1
    except Exception as e:
        logger.error(f"Error during Swedish data collection: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 