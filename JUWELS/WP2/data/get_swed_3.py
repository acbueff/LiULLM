#!/usr/bin/env python3
import argparse
from datatrove.pipeline.readers import ParquetReader

def main(min_bytes, max_bytes, output_path):
    total_bytes = 0
    doc_count = 0
    # Create a reader for the Swedish dataset (assumed identifier "swe_Latn")
    data_reader = ParquetReader("hf://datasets/HuggingFaceFW/fineweb-2/data/swe_Latn/train")
    
    with open(output_path, "w", encoding="utf-8") as outfile:
        for document in data_reader():
            text = document.text
            doc_bytes = len(text.encode("utf-8"))
            
            # If total is below the minimum, add the document unconditionally.
            if total_bytes < min_bytes:
                outfile.write(text + "\n")
                total_bytes += doc_bytes
                doc_count += 1
            else:
                # We have at least the minimum. Only add if it doesn't exceed the maximum.
                if total_bytes + doc_bytes <= max_bytes:
                    outfile.write(text + "\n")
                    total_bytes += doc_bytes
                    doc_count += 1
                # Otherwise, skip this document and continue scanning for one that fits.
            
            # Optionally, log progress every 1000 documents.
            if doc_count % 1000 == 0:
                print(f"Downloaded {doc_count} documents, {total_bytes/1024**3:.2f} GB so far...")
            
            # If we have exactly reached (or very nearly reached) the max, stop.
            if total_bytes >= max_bytes:
                print("Reached the maximum target data size.")
                break

    if total_bytes < min_bytes:
        print("Warning: The downloaded data is less than the desired minimum size.")
    print(f"Finished downloading {doc_count} documents totaling {total_bytes/1024**3:.2f} GB to {output_path}.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download Swedish data from the FineWeb2 dataset using datatrove."
    )
    parser.add_argument(
        "--min-gb",
        type=float,
        default=2.5,
        help="Minimum amount of data to download in GB (default: 2.5GB)."
    )
    parser.add_argument(
        "--max-gb",
        type=float,
        default=3.0,
        help="Maximum amount of data to download in GB (default: 3.0GB)."
    )
    parser.add_argument(
        "--output",
        type=str,
        default="swedish_data.txt",
        help="Output file to store downloaded text."
    )
    args = parser.parse_args(['--output', 'JUWELS/WP2/data/fineweb2/swedish_parquet_output_3/swedish_data.parquat'])
    min_bytes = args.min_gb * 1024**3
    max_bytes = args.max_gb * 1024**3
    if min_bytes > max_bytes:
        raise ValueError("Minimum GB cannot be greater than maximum GB.")
    main(min_bytes, max_bytes, args.output)
