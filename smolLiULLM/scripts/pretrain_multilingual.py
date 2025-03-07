#!/usr/bin/env python3
"""
Multilingual pretraining script for training a language model across multiple languages
and domains (English, Swedish, and code) with specialized tracking and checkpointing.
"""

import os
import sys
import argparse
import logging
import json
import glob
import math
import torch
import numpy as np
import wandb
from typing import Dict, Any, List, Tuple
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ""))

from src.utils.config import load_config, get_config_with_cli_overrides
from src.utils.logging import setup_logging, log_config
from src.models.llama_model import create_model, create_model_config, save_model
from src.training.trainer import train_model, set_seed
from src.data.data_utils import load_jsonl, count_tokens

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Pretrain multilingual language model")
    
    parser.add_argument(
        "--config", 
        type=str, 
        default="configs/pretrain_config.yaml",
        help="Path to pretraining configuration file"
    )
    parser.add_argument(
        "--model_config", 
        type=str, 
        default="configs/model_config.yaml",
        help="Path to model configuration file"
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default=None,
        help="Directory to save model checkpoints (overrides config)"
    )
    parser.add_argument(
        "--data_root", 
        type=str, 
        default="data/processed",
        help="Root directory with training data subdirectories"
    )
    parser.add_argument(
        "--tokenizer_path", 
        type=str, 
        default=None,
        help="Path to tokenizer (overrides config)"
    )
    parser.add_argument(
        "--resume_from_checkpoint", 
        type=str, 
        default=None,
        help="Path to checkpoint to resume training from (overrides config)"
    )
    parser.add_argument(
        "--seed", 
        type=int, 
        default=None,
        help="Random seed (overrides config)"
    )
    parser.add_argument(
        "--log_dir", 
        type=str, 
        default="outputs/logs",
        help="Directory to save logs"
    )
    parser.add_argument(
        "--wandb_project", 
        type=str, 
        default="multilingual-llm",
        help="Weights & Biases project name"
    )
    parser.add_argument(
        "--wandb_entity", 
        type=str, 
        default=None,
        help="Weights & Biases entity name"
    )
    parser.add_argument(
        "--skip_data_analysis", 
        action="store_true",
        help="Skip data analysis and use config values directly"
    )
    parser.add_argument(
        "--debug", 
        action="store_true",
        help="Enable debug mode with more verbose logging"
    )
    
    return parser.parse_args()

def calculate_token_counts(data_root: str, tokenizer_path: str, logger: logging.Logger) -> Dict[str, int]:
    """
    Calculate token counts for each dataset type in the data root directory.
    
    Args:
        data_root: Root directory containing subdirectories for each dataset type
        tokenizer_path: Path to the tokenizer to use for counting tokens
        logger: Logger instance
        
    Returns:
        Dictionary mapping dataset types to token counts
    """
    token_counts = {}
    dataset_types = ["english", "swedish", "code"]
    
    # Import tokenizer here to avoid circular imports
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    
    for dataset_type in dataset_types:
        dataset_dir = os.path.join(data_root, dataset_type)
        if not os.path.exists(dataset_dir):
            logger.warning(f"Dataset directory {dataset_dir} does not exist. Skipping.")
            continue
            
        train_file = os.path.join(dataset_dir, "train.jsonl")
        if not os.path.exists(train_file):
            logger.warning(f"Training file {train_file} does not exist. Skipping.")
            continue
            
        logger.info(f"Counting tokens in {train_file}...")
        
        # Load the dataset and count tokens
        total_tokens = 0
        samples = load_jsonl(train_file)
        
        for sample in tqdm(samples, desc=f"Counting tokens in {dataset_type}"):
            if 'text' in sample:
                tokens = tokenizer(sample['text'], return_tensors="pt")
                total_tokens += tokens.input_ids.size(1)
            elif 'content' in sample:
                tokens = tokenizer(sample['content'], return_tensors="pt")
                total_tokens += tokens.input_ids.size(1)
            else:
                logger.warning(f"Sample in {train_file} does not contain 'text' or 'content' field.")
        
        token_counts[dataset_type] = total_tokens
        logger.info(f"{dataset_type} dataset contains {total_tokens:,} tokens")
    
    return token_counts

def apply_chinchilla_scaling(total_tokens: int, logger: logging.Logger) -> Tuple[int, int]:
    """
    Apply Chinchilla scaling law to determine optimal model size.
    
    The Chinchilla paper suggests that for N tokens of training data,
    the optimal model size is approximately:
    - Parameter count: N/20
    - Model dimensions can be derived from parameter count
    
    Args:
        total_tokens: Total number of tokens across all datasets
        logger: Logger instance
        
    Returns:
        Tuple of (model_size, training_tokens)
    """
    # Chinchilla scaling law: compute optimal model size based on token count
    optimal_params = total_tokens / 20
    
    # Round to nearest power of 2 for practical implementation
    log2_params = math.log2(optimal_params)
    rounded_log2_params = round(log2_params)
    rounded_params = 2 ** rounded_log2_params
    
    # Calculate approximate model dimensions (assuming Transformer architecture)
    # For LLaMA-style models, num_params ≈ 12 * d_model^2
    # Therefore d_model ≈ sqrt(num_params / 12)
    approx_d_model = int(math.sqrt(rounded_params / 12))
    # Round to nearest multiple of 128 (typical for transformer models)
    d_model = round(approx_d_model / 128) * 128
    
    # Optimal training tokens according to Chinchilla is approximately 20x parameters
    optimal_training_tokens = rounded_params * 20
    
    logger.info(f"Chinchilla scaling law results:")
    logger.info(f"  Total available tokens: {total_tokens:,}")
    logger.info(f"  Optimal parameter count: {rounded_params:,}")
    logger.info(f"  Approximate model dimension (d_model): {d_model}")
    logger.info(f"  Optimal training tokens: {optimal_training_tokens:,}")
    
    return rounded_params, optimal_training_tokens

def create_multi_dataset_config(data_root: str, token_counts: Dict[str, int], 
                               train_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Create a configuration for multi-dataset training based on token counts.
    
    Args:
        data_root: Root directory containing subdirectories for each dataset type
        token_counts: Dictionary mapping dataset types to token counts
        train_config: Base training configuration
        
    Returns:
        Updated training configuration with multi-dataset settings
    """
    # Deep copy the train config to avoid modifying the original
    import copy
    config = copy.deepcopy(train_config)
    
    # Add multi-dataset configuration
    config['datasets'] = {
        'paths': {},
        'sampling_weights': {},
        'token_counts': token_counts
    }
    
    dataset_types = ["english", "swedish", "code"]
    total_tokens = sum(token_counts.values())
    
    for dataset_type in dataset_types:
        if dataset_type in token_counts:
            # Set dataset paths
            config['datasets']['paths'][dataset_type] = {
                'train': os.path.join(data_root, dataset_type, "train.jsonl"),
                'validation': os.path.join(data_root, dataset_type, "validation.jsonl")
            }
            
            # Calculate sampling weights based on token counts (square root scaling for better balance)
            # This gives more weight to smaller datasets but still respects relative sizes
            weight = math.sqrt(token_counts[dataset_type] / total_tokens)
            config['datasets']['sampling_weights'][dataset_type] = weight
    
    # Normalize sampling weights
    total_weight = sum(config['datasets']['sampling_weights'].values())
    for dataset_type in config['datasets']['sampling_weights']:
        config['datasets']['sampling_weights'][dataset_type] /= total_weight
    
    return config

class MultilingualDataModule:
    """Data module for loading and processing multilingual datasets."""
    
    def __init__(self, config: Dict[str, Any], tokenizer):
        self.config = config
        self.tokenizer = tokenizer
        self.datasets = {}
        self.sampling_weights = config['datasets']['sampling_weights']
        self.current_dataset_type = None
        self.max_length = config['training'].get('max_seq_length', 512)
        
    def load_datasets(self):
        """Load all datasets based on configuration."""
        from torch.utils.data import Dataset
        import random
        
        class TextDataset(Dataset):
            def __init__(self, data, tokenizer, max_length):
                self.data = data
                self.tokenizer = tokenizer
                self.max_length = max_length
                
            def __len__(self):
                return len(self.data)
                
            def __getitem__(self, idx):
                item = self.data[idx]
                text = item.get('text', item.get('content', ''))
                
                # Tokenize text
                tokens = self.tokenizer(
                    text,
                    max_length=self.max_length,
                    padding="max_length",
                    truncation=True,
                    return_tensors="pt"
                )
                
                return {
                    'input_ids': tokens.input_ids[0],
                    'attention_mask': tokens.attention_mask[0]
                }
        
        # Load all datasets
        for dataset_type, paths in self.config['datasets']['paths'].items():
            train_data = load_jsonl(paths['train'])
            val_data = load_jsonl(paths['validation'])
            
            self.datasets[dataset_type] = {
                'train': TextDataset(train_data, self.tokenizer, self.max_length),
                'validation': TextDataset(val_data, self.tokenizer, self.max_length)
            }
    
    def get_dataloader(self, split='train', dataset_type=None):
        """Get a dataloader for a specific dataset type and split."""
        from torch.utils.data import DataLoader
        
        if dataset_type is None:
            # If no specific dataset is requested, use weighted sampling across all datasets
            return self._get_weighted_dataloader(split)
            
        if dataset_type not in self.datasets:
            raise ValueError(f"Dataset type {dataset_type} not found.")
            
        dataset = self.datasets[dataset_type][split]
        
        batch_size = self.config['training'][f'per_device_{split}_batch_size']
        shuffle = split == 'train'
        
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=self.config['training'].get('dataloader_num_workers', 4)
        )
    
    def _get_weighted_dataloader(self, split='train'):
        """
        Create a dataloader that samples from all datasets based on weights.
        This is a simplification - in a full implementation, you would create a custom
        sampler to correctly handle weighted sampling across datasets.
        """
        from torch.utils.data import DataLoader, ConcatDataset
        import numpy as np
        
        # Simplified approach: concatenate all datasets
        all_datasets = [self.datasets[dtype][split] for dtype in self.datasets]
        combined_dataset = ConcatDataset(all_datasets)
        
        batch_size = self.config['training'][f'per_device_{split}_batch_size']
        shuffle = split == 'train'
        
        # In a real implementation, you'd use a custom sampler to handle the weighted sampling
        # For now, we'll just use this as a placeholder
        return DataLoader(
            combined_dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=self.config['training'].get('dataloader_num_workers', 4)
        )

class MultilingualTrainer:
    """Trainer class for multilingual model training."""
    
    def __init__(self, model, tokenizer, config, local_rank=-1):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.local_rank = local_rank
        self.is_main_process = local_rank in [-1, 0]
        self.data_module = MultilingualDataModule(config, tokenizer)
        self.logger = logging.getLogger(__name__)
        
        # Set up optimizer
        self.optimizer = self._create_optimizer()
        
        # Initialize trackers for performance monitoring
        self.best_val_losses = {
            'english': float('inf'),
            'swedish': float('inf'),
            'code': float('inf'),
            'combined': float('inf')
        }
        
        # Set up wandb if enabled
        self.use_wandb = config['training'].get('use_wandb', False)
        if self.use_wandb and self.is_main_process:
            self._init_wandb()
    
    def _create_optimizer(self):
        """Create optimizer for training."""
        from torch.optim import AdamW
        
        # Get optimizer settings from config
        lr = self.config['training'].get('learning_rate', 5e-5)
        weight_decay = self.config['training'].get('weight_decay', 0.01)
        
        # Create optimizer
        return AdamW(
            self.model.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )
    
    def _init_wandb(self):
        """Initialize Weights & Biases tracking."""
        wandb_config = {
            'model_config': self.config['model'],
            'training_config': self.config['training'],
            'datasets': {
                type_: {
                    'token_count': self.config['datasets']['token_counts'].get(type_, 0),
                    'sampling_weight': self.config['datasets']['sampling_weights'].get(type_, 0)
                }
                for type_ in self.config['datasets']['paths']
            }
        }
        
        run_name = self.config['training'].get('run_name', f"multilingual-llm-{wandb.util.generate_id()}")
        
        wandb.init(
            project=self.config['training'].get('wandb_project', 'multilingual-llm'),
            entity=self.config['training'].get('wandb_entity', None),
            config=wandb_config,
            name=run_name
        )
        
        # Define custom charts for tracking per-language performance
        wandb.define_metric("train/english_loss", summary="min")
        wandb.define_metric("train/swedish_loss", summary="min")
        wandb.define_metric("train/code_loss", summary="min")
        wandb.define_metric("val/english_loss", summary="min")
        wandb.define_metric("val/swedish_loss", summary="min")
        wandb.define_metric("val/code_loss", summary="min")
    
    def train(self):
        """Run the training loop."""
        # Load datasets
        self.logger.info("Loading datasets...")
        self.data_module.load_datasets()
        
        # Get training settings
        num_epochs = self.config['training'].get('num_train_epochs', 3)
        gradient_accumulation_steps = self.config['training'].get('gradient_accumulation_steps', 1)
        save_steps = self.config['training'].get('save_steps', 500)
        eval_steps = self.config['training'].get('eval_steps', 500)
        
        # Calculate total steps
        total_train_batch_size = (
            self.config['training']['per_device_train_batch_size'] * 
            gradient_accumulation_steps * 
            (1 if self.local_rank == -1 else torch.distributed.get_world_size())
        )
        
        # Create dataloaders for each dataset type
        train_dataloader = self.data_module.get_dataloader('train')
        val_dataloaders = {
            dtype: self.data_module.get_dataloader('validation', dtype)
            for dtype in self.config['datasets']['paths']
        }
        
        total_steps = len(train_dataloader) * num_epochs // gradient_accumulation_steps
        
        self.logger.info(f"Starting training for {num_epochs} epochs ({total_steps} steps)")
        self.logger.info(f"Total train batch size: {total_train_batch_size}")
        
        # Set up learning rate scheduler
        from transformers import get_linear_schedule_with_warmup
        warmup_steps = int(total_steps * self.config['training'].get('warmup_ratio', 0.1))
        lr_scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )
        
        # Move model to device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)
        
        # Set up progress tracking
        global_step = 0
        
        # Training loop
        for epoch in range(num_epochs):
            self.logger.info(f"Starting epoch {epoch+1}/{num_epochs}")
            
            # Set model to training mode
            self.model.train()
            
            # Reset metrics for this epoch
            epoch_loss = 0
            step_loss = 0
            
            # Create progress bar
            progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}")
            
            for step, batch in enumerate(progress_bar):
                # Move batch to device
                batch = {k: v.to(device) for k, v in batch.items()}
                
                # Forward pass
                outputs = self.model(
                    input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask'],
                    labels=batch['input_ids']
                )
                
                loss = outputs.loss
                
                # Scale loss for gradient accumulation
                loss = loss / gradient_accumulation_steps
                loss.backward()
                
                step_loss += loss.item()
                
                # Update weights if gradient accumulation is complete
                if (step + 1) % gradient_accumulation_steps == 0:
                    self.optimizer.step()
                    lr_scheduler.step()
                    self.optimizer.zero_grad()
                    
                    global_step += 1
                    epoch_loss += step_loss
                    
                    # Update progress bar
                    progress_bar.set_postfix({
                        'loss': step_loss * gradient_accumulation_steps
                    })
                    step_loss = 0
                    
                    # Log to wandb
                    if self.use_wandb and self.is_main_process and global_step % 10 == 0:
                        wandb.log({
                            "train/loss": loss.item() * gradient_accumulation_steps,
                            "train/lr": lr_scheduler.get_last_lr()[0],
                            "train/step": global_step
                        })
                    
                    # Evaluate if needed
                    if global_step % eval_steps == 0:
                        val_metrics = self.evaluate(val_dataloaders)
                        
                        # Log validation metrics
                        if self.use_wandb and self.is_main_process:
                            wandb.log({**val_metrics, "val/step": global_step})
                        
                        # Check for decreased performance and save checkpoint if needed
                        self._check_and_save_checkpoint(val_metrics, global_step)
                    
                    # Save regular checkpoint if needed
                    if global_step % save_steps == 0:
                        self._save_checkpoint(global_step)
            
            # End of epoch - log epoch metrics
            epoch_loss = epoch_loss / (len(train_dataloader) // gradient_accumulation_steps)
            self.logger.info(f"Epoch {epoch+1} completed with average loss: {epoch_loss:.4f}")
            
            # Perform full validation at the end of each epoch
            val_metrics = self.evaluate(val_dataloaders)
            if self.use_wandb and self.is_main_process:
                wandb.log({
                    **val_metrics,
                    "val/epoch": epoch + 1,
                    "val/step": global_step
                })
            
            # Save checkpoint at the end of each epoch
            self._save_checkpoint(global_step, is_epoch_end=True, epoch=epoch+1)
        
        # Save final model
        self._save_checkpoint(global_step, is_final=True)
        
        # Close wandb if used
        if self.use_wandb and self.is_main_process:
            wandb.finish()
        
        return self.model
    
    def evaluate(self, val_dataloaders):
        """Evaluate the model on validation datasets."""
        self.model.eval()
        device = next(self.model.parameters()).device
        
        metrics = {}
        combined_loss = 0
        combined_count = 0
        
        # Evaluate on each dataset type
        for dataset_type, dataloader in val_dataloaders.items():
            total_loss = 0
            total_samples = 0
            
            with torch.no_grad():
                for batch in tqdm(dataloader, desc=f"Validating {dataset_type}"):
                    # Move batch to device
                    batch = {k: v.to(device) for k, v in batch.items()}
                    
                    # Forward pass
                    outputs = self.model(
                        input_ids=batch['input_ids'],
                        attention_mask=batch['attention_mask'],
                        labels=batch['input_ids']
                    )
                    
                    loss = outputs.loss
                    
                    # Update totals
                    batch_size = batch['input_ids'].size(0)
                    total_loss += loss.item() * batch_size
                    total_samples += batch_size
            
            # Calculate average loss for this dataset
            avg_loss = total_loss / total_samples
            metrics[f"val/{dataset_type}_loss"] = avg_loss
            
            # Update combined metrics
            combined_loss += total_loss
            combined_count += total_samples
            
            self.logger.info(f"Validation {dataset_type} loss: {avg_loss:.4f}")
        
        # Calculate combined average loss
        avg_combined_loss = combined_loss / combined_count
        metrics["val/combined_loss"] = avg_combined_loss
        
        self.logger.info(f"Validation combined loss: {avg_combined_loss:.4f}")
        
        # Set model back to training mode
        self.model.train()
        
        return metrics
    
    def _check_and_save_checkpoint(self, val_metrics, global_step):
        """Check for decreased performance and save checkpoint if needed."""
        for dataset_type in self.best_val_losses:
            if dataset_type == 'combined':
                metric_key = "val/combined_loss"
            else:
                metric_key = f"val/{dataset_type}_loss"
            
            if metric_key in val_metrics:
                current_loss = val_metrics[metric_key]
                
                # Check if performance improved
                if current_loss < self.best_val_losses[dataset_type]:
                    self.best_val_losses[dataset_type] = current_loss
                    
                    # Save best checkpoint for this dataset
                    self._save_checkpoint(
                        global_step, 
                        is_best=True, 
                        dataset_type=dataset_type
                    )
                    
                    self.logger.info(f"New best {dataset_type} validation loss: {current_loss:.4f}")
                
                # Check if performance decreased significantly (more than 5%)
                elif current_loss > self.best_val_losses[dataset_type] * 1.05:
                    # Save checkpoint when performance decreases
                    self._save_checkpoint(
                        global_step, 
                        is_decrease=True, 
                        dataset_type=dataset_type
                    )
                    
                    self.logger.info(
                        f"Performance decreased for {dataset_type}: "
                        f"{self.best_val_losses[dataset_type]:.4f} -> {current_loss:.4f}"
                    )
    
    def _save_checkpoint(self, global_step, is_best=False, is_decrease=False, 
                       is_epoch_end=False, is_final=False, dataset_type=None, epoch=None):
        """Save a model checkpoint."""
        if not self.is_main_process:
            return
            
        output_dir = self.config['training']['output_dir']
        
        # Determine checkpoint name
        if is_final:
            checkpoint_dir = os.path.join(output_dir, "final")
        elif is_best and dataset_type:
            checkpoint_dir = os.path.join(output_dir, f"best_{dataset_type}")
        elif is_decrease and dataset_type:
            checkpoint_dir = os.path.join(output_dir, f"decrease_{dataset_type}_{global_step}")
        elif is_epoch_end and epoch is not None:
            checkpoint_dir = os.path.join(output_dir, f"epoch_{epoch}")
        else:
            checkpoint_dir = os.path.join(output_dir, f"step_{global_step}")
        
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Save model
        self.model.save_pretrained(checkpoint_dir)
        self.tokenizer.save_pretrained(checkpoint_dir)
        
        # Save optimizer state
        torch.save(self.optimizer.state_dict(), os.path.join(checkpoint_dir, "optimizer.pt"))
        
        # Save training state
        training_state = {
            'global_step': global_step,
            'best_val_losses': self.best_val_losses
        }
        torch.save(training_state, os.path.join(checkpoint_dir, "training_state.pt"))
        
        self.logger.info(f"Saved checkpoint to {checkpoint_dir}")

def main():
    """Main function to run multilingual pretraining."""
    # Parse arguments
    args = parse_args()
    
    # Set up logging
    log_level = logging.DEBUG if args.debug else logging.INFO
    logger = setup_logging(
        log_dir=args.log_dir,
        log_level=log_level,
        experiment_name="multilingual_pretraining"
    )
    
    try:
        # Load configurations
        logger.info(f"Loading pretraining configuration from {args.config}")
        train_config = load_config(args.config)
        
        logger.info(f"Loading model configuration from {args.model_config}")
        model_config = load_config(args.model_config)
        
        # Override configs with CLI arguments if provided
        if args.output_dir:
            train_config['output_dir'] = args.output_dir
            os.makedirs(args.output_dir, exist_ok=True)
            logger.info(f"Using output directory: {args.output_dir}")
        
        if args.tokenizer_path:
            train_config['tokenizer_path'] = args.tokenizer_path
            logger.info(f"Using tokenizer from: {args.tokenizer_path}")
        
        if args.resume_from_checkpoint:
            train_config['resume_from_checkpoint'] = args.resume_from_checkpoint
            logger.info(f"Resuming from checkpoint: {args.resume_from_checkpoint}")
        
        if args.seed is not None:
            train_config['seed'] = args.seed
            logger.info(f"Using random seed: {args.seed}")
        
        # Setup W&B configuration
        train_config['use_wandb'] = True
        train_config['wandb_project'] = args.wandb_project
        train_config['wandb_entity'] = args.wandb_entity
        logger.info("Weights & Biases logging enabled")
        
        # Set random seed for reproducibility
        set_seed(train_config.get('seed', 42))
        
        # Analyze datasets and calculate token counts (if not skipped)
        if not args.skip_data_analysis:
            logger.info("Analyzing datasets to count tokens...")
            token_counts = calculate_token_counts(
                args.data_root, 
                train_config['tokenizer_path'],
                logger
            )
            
            # Apply Chinchilla scaling law to determine optimal model size
            logger.info("Applying Chinchilla scaling law...")
            total_tokens = sum(token_counts.values())
            model_size, training_tokens = apply_chinchilla_scaling(total_tokens, logger)
            
            # Update model config based on scaling law results if needed
            # This would depend on how your model_config is structured
            
            # Create multi-dataset configuration
            train_config = create_multi_dataset_config(args.data_root, token_counts, train_config)
        else:
            logger.info("Skipping data analysis, using configuration as provided")
        
        # Log configurations
        log_config(train_config, "Training Configuration")
        log_config(model_config, "Model Configuration")
        
        # Create model
        logger.info("Creating model...")
        model_configuration = create_model_config(model_config)
        model, tokenizer = create_model(
            config=model_configuration,
            pretrained=train_config.get('resume_from_checkpoint', None)
        )
        
        # Create trainer and start training
        logger.info("Starting multilingual training...")
        trainer = MultilingualTrainer(
            model=model,
            tokenizer=tokenizer,
            config={
                'model': model_config,
                'training': train_config,
                'datasets': train_config['datasets']
            }
        )
        
        trained_model = trainer.train()
        
        logger.info("Multilingual pretraining completed successfully!")
        
    except Exception as e:
        logger.error(f"Error in multilingual pretraining: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main() 