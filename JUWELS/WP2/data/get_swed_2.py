#!/usr/bin/env python
"""
Download Swedish text data from FineWeb2 using datatrove.

This script downloads between 2.5GB and 3GB of Swedish text data from the FineWeb2 dataset
on Hugging Face. It uses a custom filter to track the size of downloaded data and stops 
once it reaches the target size range.

Usage:
    python download_swedish_fineweb.py

The downloaded data will be saved to the output directory specified in the script.
"""

# Import necessary modules
import os
import time
import logging
from typing import Optional
from datatrove.executor import LocalPipelineExecutor
from datatrove.pipeline.readers import ParquetReader
from datatrove.pipeline.writers import ParquetWriter
from datatrove.pipeline.filters import LambdaFilter

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configure environment variables for better HF Hub performance
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"    # Enable optimized transfer protocol
os.environ["HF_HUB_TIMEOUT"] = "600"              # 10 minutes timeout
os.environ["HF_HUB_DOWNLOAD_RETRY_COUNT"] = "5"   # Retry 5 times

class SizeTrackingFilter:
    """
    Filter that tracks the accumulated size of text data and stops accepting documents
    once the size limit is reached.
    """
    
    def __init__(self, min_size_gb: float = 2.5, max_size_gb: float = 3.0):
        """
        Initialize the size tracking filter.
        
        Args:
            min_size_gb: Minimum size in GB to collect before stopping
            max_size_gb: Maximum size in GB to collect
        """
        self.min_size_bytes = int(min_size_gb * 1024**3)
        self.max_size_bytes = int(max_size_gb * 1024**3)
        self.accumulated_bytes = 0
        self.document_count = 0
        self.start_time = time.time()
        self.last_log_time = time.time()
        self.log_interval = 10  # Log progress every 10 seconds
        
        logger.info(f"Initialized size tracking filter with target range: {min_size_gb}GB - {max_size_gb}GB")
    
    def __call__(self, doc, *args, **kwargs) -> bool:
        """
        Process each document, track its size, and determine whether to keep it.
        
        Returns:
            bool: True if the document should be kept, False otherwise
        """
        # Extract document text
        try:
            if hasattr(doc, 'text'):
                text = doc.text
            else:
                text = doc.get('text', '')
                
            # Calculate document size in bytes (UTF-8 encoded)
            doc_size = len(text.encode('utf-8'))
            
            # Update counters
            self.accumulated_bytes += doc_size
            self.document_count += 1
            
            # Log progress periodically
            current_time = time.time()
            if current_time - self.last_log_time > self.log_interval:
                elapsed = current_time - self.start_time
                gb_accumulated = self.accumulated_bytes / (1024**3)
                docs_per_sec = self.document_count / elapsed if elapsed > 0 else 0
                
                logger.info(f"Processed {self.document_count} documents | "
                           f"Size: {gb_accumulated:.2f}GB | "
                           f"Rate: {docs_per_sec:.1f} docs/sec")
                self.last_log_time = current_time
            
            # If we've exceeded the maximum size, reject further documents
            if self.accumulated_bytes > self.max_size_bytes:
                logger.info(f"Maximum size limit reached: {self.accumulated_bytes/(1024**3):.2f}GB > "
                           f"{self.max_size_bytes/(1024**3):.2f}GB. Stopping.")
                return False
                
            # Accept the document
            return True
            
        except Exception as e:
            logger.error(f"Error processing document: {e}")
            return False
    
    def get_stats(self) -> dict:
        """
        Get statistics about the processed data.
        
        Returns:
            dict: Statistics including document count, size, and processing rate
        """
        elapsed = time.time() - self.start_time
        return {
            "document_count": self.document_count,
            "total_size_bytes": self.accumulated_bytes,
            "total_size_gb": self.accumulated_bytes / (1024**3),
            "elapsed_seconds": elapsed,
            "documents_per_second": self.document_count / elapsed if elapsed > 0 else 0
        }

def download_swedish_fineweb(
    output_path: str,
    min_size_gb: float = 2.5,
    max_size_gb: float = 3.0,
    tasks: int = 2,
    limit: Optional[int] = None
) -> dict:
    """
    Download Swedish text data from FineWeb2 with size control.
    
    Args:
        output_path: Path where the output Parquet files will be saved
        min_size_gb: Minimum amount of data to download in GB
        max_size_gb: Maximum amount of data to download in GB
        tasks: Number of parallel download tasks
        limit: Optional limit on the number of documents to process
    
    Returns:
        dict: Statistics about the download process
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_path, exist_ok=True)
    
    # Configure the input path for Swedish data
    input_path = "hf://datasets/HuggingFaceFW/fineweb-2/data/swe_Latn/train"
    
    # Create an instance of the size tracking filter
    size_tracker = SizeTrackingFilter(min_size_gb=min_size_gb, max_size_gb=max_size_gb)
    
    # Define the pipeline
    pipeline_executor = LocalPipelineExecutor(
        pipeline=[
            ParquetReader(input_path, limit=limit),
            LambdaFilter(size_tracker),
            ParquetWriter(output_path)
        ],
        tasks=tasks  # Number of parallel tasks
    )
    
    # Run the pipeline
    logger.info(f"Starting Swedish FineWeb2 download pipeline")
    logger.info(f"Target size range: {min_size_gb}GB - {max_size_gb}GB")
    logger.info(f"Output path: {output_path}")
    
    try:
        pipeline_executor.run()
        logger.info("Pipeline completed successfully")
    except Exception as e:
        logger.error(f"Pipeline error: {e}")
        
    # Get and return statistics
    stats = size_tracker.get_stats()
    
    logger.info("=" * 50)
    logger.info("DOWNLOAD SUMMARY")
    logger.info("=" * 50)
    logger.info(f"Documents processed: {stats['document_count']:,}")
    logger.info(f"Total size: {stats['total_size_gb']:.2f}GB")
    logger.info(f"Elapsed time: {stats['elapsed_seconds']:.1f} seconds")
    logger.info(f"Processing rate: {stats['documents_per_second']:.1f} docs/sec")
    logger.info("=" * 50)
    
    return stats

if __name__ == "__main__":
    # Configuration
    OUTPUT_PATH = "JUWELS/WP2/data/fineweb2/swedish_parquet_output_2"  # Output directory
    MIN_SIZE_GB = 2.5  # Minimum size in GB
    MAX_SIZE_GB = 3.0  # Maximum size in GB
    TASKS = 2          # Number of parallel tasks
    
    # Run the download
    download_swedish_fineweb(
        output_path=OUTPUT_PATH,
        min_size_gb=MIN_SIZE_GB,
        max_size_gb=MAX_SIZE_GB,
        tasks=TASKS
    )