from datasets import load_dataset
import os

# Optionally, specify a cache directory that points to the desired folder.
cache_dir = os.path.join("JUWELS", "WP2", "data", "icelandic-cc")

# This downloads the full dataset without any file size limit.
ds = load_dataset("mideind/icelandic-common-crawl-corpus-IC3", cache_dir=cache_dir)

print("Dataset downloaded successfully:")
print(ds)