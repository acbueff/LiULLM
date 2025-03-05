#!/usr/bin/env python
"""
Wrapper script to download Swedish data directly from FineWeb-2 using Hugging Face API.
This bypasses datatrove and uses direct API access with optimized network settings.
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
    parser = argparse.ArgumentParser(description="Download Swedish data from FineWeb-2")
    
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
    parser.add_argument(
        "--max_files",
        type=int,
        default=None,
        help="Maximum number of files to process (for testing)"
    )
    parser.add_argument(
        "--token",
        type=str,
        default=None,
        help="Hugging Face token for API access"
    )
    
    return parser.parse_args()

def main():
    """Main function to run Swedish data collection using direct HF API."""
    args = parse_args()
    
    # Set token in environment if provided
    if args.token:
        os.environ["HF_TOKEN"] = args.token
        logger.info("Using provided Hugging Face token")
    
    try:
        # Get the path to the direct download script
        script_dir = os.path.dirname(os.path.abspath(__file__))
        direct_script_path = os.path.abspath(os.path.join(
            script_dir, "..", "data_download", "download_swedish_direct.py"
        ))
        
        if not os.path.exists(direct_script_path):
            logger.error(f"Direct download script not found at: {direct_script_path}")
            return 1
            
        logger.info(f"Loading direct download module from: {direct_script_path}")
        
        # Load the module from the file path
        direct_module = load_module_from_path("download_swedish_direct", direct_script_path)
        
        # Configure HTTP session
        direct_module.configure_http_session()
        
        # Run the direct download
        logger.info(f"Starting Swedish data collection using direct HF API")
        logger.info(f"Output directory: {args.output_dir}")
        logger.info(f"Size limit: {args.size_limit} GB")
        if args.max_files:
            logger.info(f"Processing maximum {args.max_files} files")
        
        # Download the data
        result_files = direct_module.download_swedish_data(
            args.output_dir, 
            args.size_limit,
            args.max_files
        )
        
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