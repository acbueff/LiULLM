#!/usr/bin/env python3
"""
Utility script to fix the logging setup in all local scripts.
This fixes the TypeError: setup_logging() got an unexpected keyword argument 'log_file' error.
"""

import os
import sys
import glob
import re

def fix_logging_in_script(script_path):
    """Fix the logging setup in a script."""
    print(f"Fixing logging in script: {script_path}")
    
    with open(script_path, 'r') as f:
        content = f.read()
    
    # Check if the script already has the fixed logging setup
    if "log_timestamp = int(start_time)" in content:
        print(f"Script {script_path} already fixed. Skipping.")
        return
    
    # Replace the old logging setup with the new one
    content = re.sub(
        r'(\s+)# Setup logging\s+log_level = logging\.DEBUG if args\.debug else logging\.INFO\s+log_file = os\.path\.join\(args\.log_dir, f"[^"]+"\)\s+setup_logging\(log_level=log_level, log_file=log_file\)',
        r'\1# Setup logging - using the correct parameters for the setup_logging function'
        r'\1log_level = logging.DEBUG if args.debug else logging.INFO'
        r'\1log_timestamp = int(start_time)'
        r'\1experiment_name = f"{script_name}_{log_timestamp}"'
        r'\1'
        r'\1# Create the log directory if it doesn\'t exist'
        r'\1os.makedirs(args.log_dir, exist_ok=True)'
        r'\1'
        r'\1# Setup logging with the correct parameters'
        r'\1setup_logging('
        r'\1    log_dir=args.log_dir,'
        r'\1    log_level=log_level,'
        r'\1    experiment_name=experiment_name'
        r'\1)',
        content
    )
    
    # Add the script_name variable definition near the start of the main function
    content = re.sub(
        r'def main\(\):\s+"""[^"]+"""\s+start_time = time\.time\(\)',
        r'def main():\n    """Main function to run the script."""\n    # Get script name without extension for logging\n    script_name = os.path.basename(__file__).replace(".py", "")\n    start_time = time.time()',
        content
    )
    
    with open(script_path, 'w') as f:
        f.write(content)
    
    print(f"Fixed logging in script: {script_path}")

def main():
    """Fix logging in all scripts in the local directory."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    scripts = glob.glob(os.path.join(script_dir, "run_*.py"))
    
    print(f"Found {len(scripts)} scripts to fix")
    
    for script in scripts:
        if script != os.path.abspath(__file__) and script != os.path.join(script_dir, "fix_all_scripts.py"):
            fix_logging_in_script(script)
    
    print("All scripts fixed!")

if __name__ == "__main__":
    main() 