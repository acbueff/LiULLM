# LiULLM Local Execution Scripts

This directory contains scripts for running the LiULLM training pipeline locally on a single machine with GPU support. These scripts provide an alternative to the Slurm-based execution used on HPC clusters.

## Available Scripts

- `run_data_collection.py` - Download pretraining or instruction tuning data
- `run_preprocessing.py` - Preprocess raw text data for training
- `run_tokenizer_training.py` - Train a tokenizer on preprocessed data
- `run_pretraining.py` - Run model pretraining on a local GPU
- `run_finetuning.py` - Run instruction fine-tuning on a local GPU
- `run_evaluation.py` - Evaluate model performance locally

## Requirements

- CUDA-compatible GPU
- Python 3.8 or higher
- PyTorch 2.0 or higher
- Transformers 4.30 or higher
- All dependencies as specified in the main LiULLM requirements

## Usage Examples

### 1. Data Collection

```bash
# Download pretraining data for English
python scripts/local/run_data_collection.py --data_type pretraining --languages en

# Download instruction tuning data
python scripts/local/run_data_collection.py --data_type instruction
```

### 2. Data Preprocessing

```bash
# Preprocess the downloaded raw data
python scripts/local/run_preprocessing.py --input_dir data/raw --output_dir data/processed
```

### 3. Tokenizer Training

```bash
# Train a tokenizer on preprocessed data
python scripts/local/run_tokenizer_training.py --input_dir data/processed --output_dir data/tokenizer --vocab_size 32000
```

### 4. Pretraining

```bash
# Run model pretraining on GPU 0 (adjust batch size for your GPU memory)
python scripts/local/run_pretraining.py --batch_size 4 --gradient_accumulation_steps 8 --gpu_id 0
```

### 5. Fine-tuning

```bash
# Run instruction fine-tuning on a pretrained model
python scripts/local/run_finetuning.py --model_path outputs/checkpoints/pretrain/final_model --batch_size 2 --gpu_id 0
```

### 6. Evaluation

```bash
# Evaluate a trained model
python scripts/local/run_evaluation.py --model_path outputs/finetuned/final_model --gpu_id 0
```

## Configuration

Each script reads from the standard configuration files in the `configs/` directory by default. Command-line arguments can be used to override specific configuration parameters.

For detailed information about each script's options, run the script with the `--help` flag:

```bash
python scripts/local/run_pretraining.py --help
```

## GPU Memory Considerations

When running on a local GPU, you may need to adjust batch sizes, sequence lengths, and model sizes to fit within your available GPU memory. The scripts include parameters to control these aspects:

- `--batch_size` - Control the batch size
- `--gradient_accumulation_steps` - Accumulate gradients over multiple batches
- `--gpu_id` - Specify which GPU to use if you have multiple

## Logs and Outputs

- Logs are saved in `outputs/logs/` by default
- Trained models are saved in `outputs/checkpoints/pretrain/` or `outputs/finetuned/`
- Evaluation results are saved in `outputs/evaluation/`

You can customize these paths using command-line arguments. 