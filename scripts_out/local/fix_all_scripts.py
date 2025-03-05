#!/usr/bin/env python3
"""
Utility script to fix import paths in all local scripts.
"""

import os
import sys
import glob
import re

def fix_script(script_path):
    """Fix import paths in a script."""
    print(f"Fixing script: {script_path}")
    
    with open(script_path, 'r') as f:
        content = f.read()
    
    # Replace old import paths with new ones
    content = re.sub(
        r'# Add LiULLM directory to path\nroot_dir = os\.path\.dirname\(os\.path\.dirname\(os\.path\.dirname\(os\.path\.abspath\(__file__\)\)\)\)\nsys\.path\.insert\(0, root_dir\)',
        '# Add current directory to path to find LiULLM\ncurrent_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))\nliullm_dir = os.path.join(current_dir, "LiULLM")\nsys.path.insert(0, liullm_dir)',
        content
    )
    
    # Replace import statements
    content = re.sub(
        r'# Import LiULLM modules.*?\nfrom src\.([^ ]+) import ([^\n]+)',
        '# Now try to import from LiULLM\ntry:\n    from LiULLM.src.\\1 import \\2\nexcept ImportError:\n    # If that fails, adjust path again to try direct import\n    sys.path.insert(0, os.path.dirname(os.path.dirname(current_dir)))\n    from src.\\1 import \\2',
        content,
        flags=re.DOTALL
    )
    
    # Replace root_dir with liullm_dir
    content = content.replace('root_dir', 'liullm_dir')
    
    # Add config path check
    if 'config = load_config(args.config)' in content:
        content = content.replace(
            'config = load_config(args.config)',
            '# Check if config file exists in current directory, if not look in LiULLM directory\n    config_path = args.config\n    if not os.path.exists(config_path):\n        config_path = os.path.join(liullm_dir, args.config)\n        logger.info(f"Config not found in current directory, trying: {config_path}")\n    \n    # Load configuration\n    config = load_config(config_path)\n    logger.info(f"Loaded configuration from: {config_path}")'
        )
    
    with open(script_path, 'w') as f:
        f.write(content)
    
    print(f"Fixed script: {script_path}")

def main():
    """Fix all scripts in the local directory."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    scripts = glob.glob(os.path.join(script_dir, "run_*.py"))
    
    print(f"Found {len(scripts)} scripts to fix")
    
    for script in scripts:
        if script != os.path.abspath(__file__):  # Don't fix this script
            fix_script(script)
    
    print("All scripts fixed!")

if __name__ == "__main__":
    main() 