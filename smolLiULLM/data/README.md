# Data Directory

The data files that were previously in this directory have been moved to:
`/home/andbu/Documents/trustllm/TechStack/data/LiULLM/smolLiULLM/data/`

This was done to prevent large files from being tracked by git.

## Data Structure
- `raw/`: Raw data files
- `processed/`: Processed training data
- `instruction_tuning/`: Instruction tuning data
- `tokenizer/`: Tokenizer files

If you need to use these files, you can access them at the new location or create symbolic links using the script in the repository root: `create_data_symlinks.sh`.
