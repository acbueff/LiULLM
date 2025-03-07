#!/usr/bin/env python
"""
Unified Data Preprocessing Pipeline

This pipeline performs:
  1. Formatting & Normalization
     - Reads raw JSONL files from WP2/data/raw.
     - Normalizes text (fixes encoding, lowercases, trims whitespace, etc.).
  2. (Optional) Quality Filtering
     - Computes basic quality signals (e.g. text length); you can add more sophisticated methods.
  3. Deduplication
     - Removes duplicate (or near‐duplicate) rows based on a text hash.
  4. Finalization
     - Writes a final parquet file ready for downstream LLM training.
     
Folders used:
  - WP2/data/raw       : raw input files (JSONL)
  - WP2/data/normalized: intermediate normalized parquet files
  - WP2/data/dedup     : intermediate deduplicated parquet files
  - WP2/data/final     : final clean data output
     
Usage:
    python pipeline.py [--scheduler SCHEDULER_FILE] [--debug]
"""

import argparse
import pathlib
import json
import logging
import hashlib
import shutil

import ftfy
import pyarrow as pa
import pyarrow.parquet as pq

import dask
import dask.dataframe as dd
from dask.distributed import Client

import polars as pl


def normalize_text(text: str) -> str:
    """
    Normalize the text:
      - Fix encoding errors using ftfy.
      - Lowercase, trim extra whitespace etc.
    """
    # First, fix mis-encoded characters and trim whitespace
    fixed = ftfy.fix_text(text.strip())
    # Additional normalization (lowercase) may be useful
    return fixed.lower()


def format_and_normalize_file(raw_file: pathlib.Path, out_file: pathlib.Path) -> pathlib.Path:
    """
    Read a raw JSONL file (assumed one JSON object per line), 
    normalize the text field, and write out as a Parquet file.
    
    The JSONL file is assumed to contain at least an "id" and "text" field.
    """
    logging.info(f"Processing raw file: {raw_file}")
    rows = []
    with raw_file.open('r', encoding='utf-8') as fh:
        for line in fh:
            try:
                data = json.loads(line)
                # Normalize text field; if key missing, use empty string
                data["text"] = normalize_text(data.get("text", ""))
                rows.append(data)
            except Exception as e:
                logging.error(f"Error processing line in {raw_file}: {e}")
    if not rows:
        logging.warning(f"No content in {raw_file}")
        return out_file
    # Convert list of dicts into a PyArrow Table (assume all rows share the same keys)
    table = pa.Table.from_pydict({k: [row.get(k) for row in rows] for k in rows[0].keys()})
    pq.write_table(table, out_file)
    logging.info(f"Wrote normalized file: {out_file}")
    return out_file


def deduplicate_parquet_files(parquet_files: list, out_path: str) -> str:
    """
    Read all normalized parquet files as a Dask DataFrame, compute a hash of the text,
    and drop duplicate rows.
    
    (This is a simplified version. In a full production system you might use the minhash dedup logic.)
    """
    logging.info("Starting deduplication step")
    # Read all parquet files
    ddf = dd.read_parquet(parquet_files)
    
    # Function to compute an SHA1 hash from text.
    def compute_hash(text):
        return hashlib.sha1(text.encode('utf-8')).hexdigest()
    
    # Create a new column 'text_hash'
    ddf["text_hash"] = ddf["text"].map(compute_hash, meta=("text_hash", str))
    # Deduplicate based on the text hash
    ddf = ddf.drop_duplicates(subset=["text_hash"])
    
    # Write the deduplicated data
    ddf.to_parquet(out_path, engine="pyarrow", write_index=False)
    logging.info(f"Deduplication complete, written output: {out_path}")
    return out_path


def compute_quality_signals(parquet_file: pathlib.Path) -> pl.DataFrame:
    """
    (Optional) Compute quality signals using Polars.
    
    For example, signal such as text length could be a proxy for filtering.
    """
    df = pl.read_parquet(parquet_file)
    # Add a simple quality signal: length of text
    df = df.with_column(pl.col("text").str.lengths().alias("text_length"))
    return df


def main(args):
    # Define directories
    base_dir = pathlib.Path("WP2/data")
    raw_dir = base_dir / "raw"
    normalized_dir = base_dir / "normalized"
    dedup_dir = base_dir / "dedup"
    final_dir = base_dir / "final"
    
    # Ensure output folders exist
    for d in [normalized_dir, dedup_dir, final_dir]:
        d.mkdir(parents=True, exist_ok=True)
    
    # STEP 1: Format & Normalize
    norm_files = []
    for raw_file in sorted(raw_dir.glob("*.jsonl")):
        norm_out = normalized_dir / (raw_file.stem + ".parquet")
        format_and_normalize_file(raw_file, norm_out)
        norm_files.append(str(norm_out))
    
    # (STEP 2: Quality Filtering could be inserted here if you want to drop low-quality samples)
    # For example, you might iterate over norm_files, compute signals, and then filter out
    # rows that have a very short text length.
    #
    # Example (optional):
    # for nf in norm_files:
    #     df = compute_quality_signals(pathlib.Path(nf))
    #     # Filter for texts with length > 30 characters
    #     df = df.filter(pl.col("text_length") > 30)
    #     df.write_parquet(nf)
    
    # STEP 3: Deduplication
    dedup_out = str(dedup_dir / "deduped.parquet")
    deduplicate_parquet_files(norm_files, dedup_out)
    
    # STEP 4: Finalize the cleaned dataset
    final_out = final_dir / "final.parquet"
    shutil.copy(dedup_out, final_out)
    logging.info(f"Final clean dataset available at: {final_out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Streamlined Data Preprocessing Pipeline")
    parser.add_argument("--scheduler", help="Path to a Dask scheduler file (optional)", default=None)
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s"
    )

    # If a scheduler file is provided, use a Dask distributed Client with that scheduler.
    if args.scheduler:
        client = Client(scheduler_file=args.scheduler)
        logging.info(f"Dask client connected to scheduler at {args.scheduler}")
    else:
        client = Client()  # local threaded client for testing

    main(args)