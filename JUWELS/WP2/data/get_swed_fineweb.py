#!/usr/bin/env python
"""
Download Swedish text data from the FineWeb2 dataset using datatrove and store it as Parquet files,
ensuring that we collect approximately 2.5GB of text data (allowing a single file to exceed the cap if needed).

Pipeline steps:
  1. Reads the Swedish subset ("swe_Latn") of the FineWeb2 dataset from Hugging Face using ParquetReader.
  2. Passes each document through a custom SizeLimitFilter callable (via LambdaFilter) that returns True only when the
     cumulative text size (UTF‑8 encoded) is below or just exceeds 2.5GB. Once the limit is reached,
     further documents are rejected.
  3. Writes the accepted documents into the output folder as Parquet files via ParquetWriter.

You can set document_limit to a specific number for testing purposes (or -1 to process the full dataset).
"""

# Import OS and set environment variables at the very beginning
import os
# Configure Hugging Face Hub to use a longer timeout for downloading files
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"  # Enable the optimized HF transfer protocol
os.environ["HF_HUB_TIMEOUT"] = "300"           # Set the timeout to 5 minutes (300 seconds)

# Now we can import the rest
from datatrove.executor import LocalPipelineExecutor
from datatrove.pipeline.readers import ParquetReader
from datatrove.pipeline.writers import ParquetWriter
from datatrove.pipeline.filters import LambdaFilter

# -----------------------------------------------------------------------------
# Custom filter implementation to ensure we stop after collecting ~2.5GB of text data.
# This filter is used in a LambdaFilter stage.
# -----------------------------------------------------------------------------
class SizeLimitFilter:
    def __init__(self, size_limit_bytes):
        self.size_limit_bytes = size_limit_bytes
        self.accumulated = 0
        self.limit_reached = False

    def __call__(self, doc, *args, **kwargs):
        # If limit was already reached in a previous call, reject further docs.
        if self.limit_reached:
            return False

        # If doc is a generator-like object, try to extract the document.
        if hasattr(doc, '__iter__') and not isinstance(doc, (list, tuple, dict, str, bytes)):
            try:
                doc = next(doc)
            except StopIteration:
                return False

        # Obtain the document text; assume field is 'text'.
        try:
            text = doc.text
        except AttributeError:
            text = doc.get("text", "")
        doc_size = len(text.encode("utf-8"))

        # If adding this document would exceed the cap, accept it (so it's complete)
        # and mark the limit as reached so no further documents are accepted.
        if self.accumulated + doc_size > self.size_limit_bytes:
            self.accumulated += doc_size
            self.limit_reached = True
            return True

        # Normal case: accept the document and update the accumulated size.
        self.accumulated += doc_size
        return True

# -----------------------------------------------------------------------------
# Pipeline configuration
# -----------------------------------------------------------------------------
# Use -1 to process all available documents.
document_limit = -1

# Input path for Swedish subset of FineWeb2 on Hugging Face.
input_path = "hf://datasets/HuggingFaceFW/fineweb-2/data/swe_Latn/train"

# Output directory where the Parquet files will be stored.
output_path = "JUWELS/WP2/data/fineweb2/swedish_parquet_output"

# 2.5 GB limit in bytes.
size_limit_bytes = int(2.5 * 1024**3)

# Create our size limiter instance.
size_limiter = SizeLimitFilter(size_limit_bytes)

# Instead of tasks=10, use tasks=2 to reduce the number of parallel connections
# This helps reduce network congestion which can lead to DNS resolution failures
pipeline_executor = LocalPipelineExecutor(
    pipeline=[
        ParquetReader(input_path, limit=document_limit),
        LambdaFilter(size_limiter),
        ParquetWriter(output_path)
    ],
    tasks=2  # Using fewer tasks to reduce network load
)

if __name__ == "__main__":
    # Let's try to manually patch Hugging Face's download mechanism with longer timeouts
    # This needs to be done at runtime just before executing the pipeline
    
    try:
        # First, let's patch urllib3 which is used by requests and huggingface_hub
        import urllib3
        # Increase the default timeout for all urllib3 connections
        urllib3.connectionpool.HTTPConnectionPool.timeout = 300
        urllib3.connectionpool.HTTPSConnectionPool.timeout = 300
        
        # Now patch requests which is used by huggingface_hub
        import requests
        from requests.adapters import HTTPAdapter
        
        # Create a custom adapter with longer timeouts and more retries
        class TimeoutAdapter(HTTPAdapter):
            def __init__(self, *args, **kwargs):
                # Default timeout of 5 minutes
                self.timeout = 300
                # Configure the adapter with more retries
                kwargs.setdefault('max_retries', urllib3.Retry(
                    total=10,  # Total number of retries
                    backoff_factor=2,  # Exponential backoff
                    status_forcelist=[408, 429, 500, 502, 503, 504],  # Retry on these status codes
                    allowed_methods=["HEAD", "GET", "OPTIONS"]  # Only retry for these methods
                ))
                super().__init__(*args, **kwargs)
            
            def send(self, request, **kwargs):
                # Always use our timeout
                kwargs.setdefault('timeout', self.timeout)
                return super().send(request, **kwargs)
        
        # Apply our adapter to the default session
        session = requests.Session()
        adapter = TimeoutAdapter()
        session.mount('http://', adapter)
        session.mount('https://', adapter)
        
        # Replace the default session that requests uses
        requests.sessions.Session = lambda: session
        
        # Also try to directly patch any huggingface_hub internal sessions
        import huggingface_hub
        try:
            # This is a speculative patch as the internal structure might vary
            if hasattr(huggingface_hub, "_get_session"):
                original_get_session = huggingface_hub._get_session
                def patched_get_session(*args, **kwargs):
                    sess = original_get_session(*args, **kwargs)
                    sess.mount('http://', TimeoutAdapter())
                    sess.mount('https://', TimeoutAdapter())
                    return sess
                huggingface_hub._get_session = patched_get_session
        except:
            # If that fails, no problem - we've already patched at lower levels
            pass
    except Exception as e:
        # If our patching fails, just print a warning and continue
        print(f"Warning: Failed to patch timeout settings: {e}")
    
    # Now run the pipeline
    pipeline_executor.run()
    print(f"Swedish text data has been downloaded and stored as Parquet files at {output_path}")