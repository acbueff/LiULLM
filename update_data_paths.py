#!/usr/bin/env python3

import os
import re
import glob
from pathlib import Path

# Define the old and new base paths
OLD_DATA_BASE = "/home/andbu/Documents/trustllm/TechStack/LiULLM"
NEW_DATA_BASE = "/home/andbu/Documents/trustllm/TechStack/data/LiULLM"

# Extensions to search
CODE_EXTENSIONS = ['.py', '.sh', '.yaml', '.yml', '.json', '.md']

# Get the repository root directory
repo_root = os.path.dirname(os.path.abspath(__file__))

# Patterns to search for
path_patterns = [
    r'(?:[\'\"]|^|\s)(/home/andbu/Documents/trustllm/TechStack/LiULLM/[^\'"]*\.(?:parquet|arrow|jsonl|json|txt))(?:[\'\"]|$|\s)',
    r'(?:[\'\"]|^|\s)(LiULLM/[^\'"]*\.(?:parquet|arrow|jsonl|json|txt))(?:[\'\"]|$|\s)',
    r'(?:[\'\"]|^|\s)(data/[^\'"]*\.(?:parquet|arrow|jsonl|json|txt))(?:[\'\"]|$|\s)',
    r'(?:[\'\"]|^|\s)(smolLiULLM/data/[^\'"]*\.(?:parquet|arrow|jsonl|json|txt))(?:[\'\"]|$|\s)',
]

def update_file(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
        
        original_content = content
        modified = False
        
        # Replace absolute paths
        for pattern in path_patterns:
            matches = re.findall(pattern, content)
            if matches:
                for match in matches:
                    if OLD_DATA_BASE in match:
                        new_path = match.replace(OLD_DATA_BASE, NEW_DATA_BASE)
                        content = content.replace(match, new_path)
                        modified = True
                        print(f"Replaced: {match} -> {new_path}")
        
        # Only write to the file if changes were made
        if modified:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"Updated file: {file_path}")
        
        return modified
    except Exception as e:
        print(f"Error processing file {file_path}: {e}")
        return False

def scan_directory():
    modified_files = 0
    
    # Find all code files
    for ext in CODE_EXTENSIONS:
        for file_path in glob.glob(f"{repo_root}/**/*{ext}", recursive=True):
            if "/.git/" not in file_path:
                if update_file(file_path):
                    modified_files += 1
    
    print(f"Total files modified: {modified_files}")

if __name__ == "__main__":
    scan_directory()
