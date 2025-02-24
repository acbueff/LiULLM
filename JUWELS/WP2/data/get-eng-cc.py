import os
import requests
from huggingface_hub import HfApi
from dask import delayed, compute

# Define download limit: 5GB in bytes
LIMIT_BYTES = 5 * 1024**3

def get_file_size(filename):
    """
    Retrieve the file size (in bytes) using a GET request with a Range header.
    Fallback to Content-Length if Content-Range is unavailable.
    """
    url = f"https://huggingface.co/datasets/PleIAs/English-PD/resolve/main/{filename}"
    headers = {"Range": "bytes=0-0"}
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
        print(f"Warning: Unable to fetch size for {filename}.")
        return 0

def download_file(filename):
    """
    Download a single file from the dataset repository and store it locally.
    """
    url = f"https://huggingface.co/datasets/PleIAs/English-PD/resolve/main/{filename}"
    local_path = os.path.join("JUWELS/WP2/data/eng-common-corpus", filename)
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(local_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
    print(f"Downloaded {filename} to {local_path}")
    return filename

def main():
    api = HfApi()
    # List all files in the PleIAs/English-PD repository.
    files = api.list_repo_files("PleIAs/English-PD", repo_type="dataset")
    print(f"Found {len(files)} files in PleIAs/English-PD.")
    
    # Use Dask delayed to concurrently get file sizes.
    size_tasks = [delayed(get_file_size)(f) for f in files]
    sizes = compute(*size_tasks)
    
    # Select files until the cumulative size reaches 5GB.
    selected_files = []
    total_size = 0
    for f, size in zip(files, sizes):
        if total_size + size <= LIMIT_BYTES:
            selected_files.append(f)
            total_size += size
        else:
            break

    print(f"Selected {len(selected_files)} files for download (total: {total_size/1024**3:.2f} GB).")
    
    # Create Dask delayed download tasks for the selected files.
    download_tasks = [delayed(download_file)(f) for f in selected_files]
    
    # Execute the download tasks concurrently.
    compute(*download_tasks)

if __name__ == "__main__":
    main()
