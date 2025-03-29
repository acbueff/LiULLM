"""
Training module for LLM pretraining and fine-tuning.
"""

import os
import torch
import logging
import math
import random
import numpy as np
from typing import Dict, Any, Optional, List, Tuple, Union, Callable
from transformers import (
    Trainer, 
    TrainingArguments,
    DataCollatorForLanguageModeling,
    EarlyStoppingCallback,
    default_data_collator,
    TrainerCallback,
    TrainerState,
    TrainerControl
)
from transformers.integrations import WandbCallback
from datasets import Dataset, load_dataset
import wandb

from ..utils.logging import log_metrics

logger = logging.getLogger(__name__)

def set_seed(seed: int):
    """
    Set random seed for reproducibility.
    
    Args:
        seed: Random seed.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        
class LossLoggingCallback(TrainerCallback):
    """Callback for logging training loss at each step."""
    
    def __init__(self, log_steps: int = 10):
        """
        Initialize callback.
        
        Args:
            log_steps: How often to log metrics (every N steps).
        """
        self.log_steps = log_steps
    
    def on_log(self, args, state: TrainerState, control: TrainerControl, logs: Dict[str, float], **kwargs):
        """Log metrics on each log event."""
        if state.global_step % self.log_steps == 0 and "loss" in logs:
            metrics = {k: v for k, v in logs.items() if not k.startswith("_")}
            log_metrics(metrics, state.global_step)
            
class WandBCustomCallback(WandbCallback):
    """Extended W&B callback for custom logging."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize with configuration.
        
        Args:
            config: Training configuration dictionary.
        """
        super().__init__()
        self.config = config
        self.training_config = config.get('training', {})
        
    def setup(self, args, state, model, **kwargs):
        """Set up W&B run with custom configuration."""
        if self._wandb is None:
            return
        
        # Extract relevant info from config
        if 'model' in self.config:
            model_info = {
                "hidden_size": self.config['model'].get('hidden_size', None),
                "num_layers": self.config['model'].get('num_hidden_layers', None),
                "num_heads": self.config['model'].get('num_attention_heads', None),
                "vocab_size": self.config['model'].get('vocab_size', None),
            }
        else:
            model_info = {}
            
        # Define custom config for W&B
        wandb_config = {
            "model": model_info,
            "batch_size": self.config.get('batch_size', {}).get('per_device_train_batch_size', None),
            "gradient_accumulation_steps": self.config.get('batch_size', {}).get('gradient_accumulation_steps', None),
            "learning_rate": self.training_config.get('learning_rate', None),
            "weight_decay": self.training_config.get('weight_decay', None),
            "warmup_steps": self.training_config.get('warmup_steps', None),
            "max_steps": self.training_config.get('max_steps', None),
            **kwargs.get("config", {})
        }
        
        # Initialize W&B run
        if not wandb.run:
            wandb.init(
                project=os.environ.get("WANDB_PROJECT", "liullm"),
                name=os.environ.get("WANDB_RUN_NAME", None),
                config=wandb_config
            )
        elif kwargs.get("config", None) is not None:
            # Update config if run already exists
            wandb.config.update(wandb_config, allow_val_change=True)

def prepare_datasets(
    tokenizer,
    train_file: str,
    validation_file: Optional[str] = None,
    block_size: int = 1024,
    preprocessing_num_workers: Optional[int] = None,
    overwrite_cache: bool = False
) -> Tuple[Dataset, Optional[Dataset]]:
    """
    Prepare tokenized datasets for language model training.
    
    Args:
        tokenizer: Tokenizer to use for encoding.
        train_file: Path to training data file.
        validation_file: Path to validation data file.
        block_size: Maximum sequence length for training.
        preprocessing_num_workers: Number of workers for preprocessing.
        overwrite_cache: Whether to overwrite cached datasets.
        
    Returns:
        Tuple of (train_dataset, val_dataset).
    """
    logger.info(f"Loading datasets from {train_file}")
    
    data_files = {"train": train_file}
    if validation_file:
        data_files["validation"] = validation_file
        
    extension = train_file.split(".")[-1]
    if extension == "txt":
        extension = "text"
        
    # Load raw datasets
    raw_datasets = load_dataset(
        extension,
        data_files=data_files,
        cache_dir=None,
        streaming=False
    )
    
    # Tokenize and chunk text
    def tokenize_function(examples):
        output = tokenizer(examples["text"])
        return output
        
    tokenized_datasets = raw_datasets.map(
        tokenize_function,
        batched=True,
        num_proc=preprocessing_num_workers,
        remove_columns=["text"],
        load_from_cache_file=not overwrite_cache,
        desc="Tokenizing datasets",
    )
    
    # Group texts into blocks of block_size
    def group_texts(examples):
        concatenated = {k: sum(examples[k], []) for k in examples.keys()}
        total_length = len(concatenated[list(examples.keys())[0]])
        
        # Drop the small remainder, and then split into chunks of block_size
        total_length = (total_length // block_size) * block_size
        
        # Split into chunks
        result = {
            k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated.items()
        }
        
        # Add labels for CLM (just the input_ids)
        result["labels"] = result["input_ids"].copy()
        return result
    
    # Group into chunks of block_size
    lm_datasets = tokenized_datasets.map(
        group_texts,
        batched=True,
        num_proc=preprocessing_num_workers,
        load_from_cache_file=not overwrite_cache,
        desc=f"Grouping texts into chunks of {block_size}",
    )
    
    train_dataset = lm_datasets["train"]
    validation_dataset = lm_datasets["validation"] if "validation" in lm_datasets else None
    
    logger.info(f"Train dataset size: {len(train_dataset)}")
    if validation_dataset:
        logger.info(f"Validation dataset size: {len(validation_dataset)}")
        
    return train_dataset, validation_dataset

def create_training_args(config: Dict[str, Any], output_dir: str) -> TrainingArguments:
    """
    Create training arguments from configuration.
    
    Args:
        config: Training configuration dictionary.
        output_dir: Directory to save outputs to.
        
    Returns:
        TrainingArguments instance.
    """
    training_config = config['training']
    batch_size_config = config['batch_size']
    checkpoint_config = config['checkpointing']
    eval_config = config['evaluation']
    logging_config = config['logging']
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Set up mixed precision training
    fp16 = config.get('mixed_precision', {}).get('enabled', False)
    bf16 = config.get('mixed_precision', {}).get('precision', 'fp16') == 'bf16'
    
    # Configure TrainingArguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        
        # Training parameters
        learning_rate=training_config.get('learning_rate', 2e-5),
        weight_decay=training_config.get('weight_decay', 0.01),
        adam_beta1=training_config.get('adam_beta1', 0.9),
        adam_beta2=training_config.get('adam_beta2', 0.999),
        adam_epsilon=training_config.get('adam_epsilon', 1e-8),
        max_grad_norm=training_config.get('max_grad_norm', 1.0),
        
        # Training schedule
        max_steps=training_config.get('max_steps', -1),
        num_train_epochs=training_config.get('num_train_epochs', 3),
        warmup_steps=training_config.get('warmup_steps', 0),
        warmup_ratio=training_config.get('warmup_ratio', 0),
        lr_scheduler_type=training_config.get('lr_scheduler_type', 'linear'),
        
        # Batch size
        per_device_train_batch_size=batch_size_config.get('per_device_train_batch_size', 8),
        per_device_eval_batch_size=batch_size_config.get('per_device_eval_batch_size', 8),
        gradient_accumulation_steps=batch_size_config.get('gradient_accumulation_steps', 1),
        
        # Mixed precision
        fp16=fp16 and not bf16,
        bf16=bf16,
        
        # Logging
        logging_dir=os.path.join(output_dir, 'logs'),
        logging_strategy="steps",
        logging_steps=logging_config.get('logging_steps', 100),
        report_to=logging_config.get('report_to', "none").split(","),
        
        # Checkpointing
        save_strategy="steps",
        save_steps=checkpoint_config.get('save_steps', 1000),
        save_total_limit=checkpoint_config.get('save_total_limit', 3),
        
        # Evaluation
        evaluation_strategy=eval_config.get('eval_strategy', "no"),
        eval_steps=eval_config.get('eval_steps', None),
        
        # Other settings
        dataloader_num_workers=4,
        group_by_length=False,
        dataloader_drop_last=False,
        seed=config.get('initialization', {}).get('random_seed', 42),
        push_to_hub=False,
    )
    
    return training_args

def train_model(
    model,
    tokenizer,
    train_dataset: Dataset,
    validation_dataset: Optional[Dataset],
    config: Dict[str, Any],
    output_dir: str,
    resume_from_checkpoint: Optional[Union[str, bool]] = None,
    local_rank: int = -1
) -> Tuple[Trainer, Dict[str, float]]:
    """
    Train an LLM model with the provided configuration.
    
    Args:
        model: Model to train.
        tokenizer: Tokenizer for the model.
        train_dataset: Training dataset.
        validation_dataset: Validation dataset (or None).
        config: Training configuration dictionary.
        output_dir: Directory to save outputs to.
        resume_from_checkpoint: Whether to resume from checkpoint, and if so, which one.
        local_rank: Local rank for distributed training.
        
    Returns:
        Tuple of (trainer, metrics).
    """
    logger.info("Setting up training...")
    
    # Set seed for reproducibility
    if 'initialization' in config and 'random_seed' in config['initialization']:
        set_seed(config['initialization']['random_seed'])
    
    # Create training arguments
    training_args = create_training_args(config, output_dir)
    
    # Data collator for language modeling
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,  # Not using masked language modeling
        pad_to_multiple_of=8 if training_args.fp16 or training_args.bf16 else None,
    )
    
    # Setup callbacks
    callbacks = [LossLoggingCallback()]
    
    # Add early stopping if configured
    if config.get('early_stopping', {}).get('enabled', False):
        early_stopping_config = config['early_stopping']
        patience = early_stopping_config.get('patience', 3)
        callbacks.append(EarlyStoppingCallback(
            early_stopping_patience=patience
        ))
    
    # Add W&B callback if enabled
    if "wandb" in training_args.report_to:
        callbacks.append(WandBCustomCallback(config))
    
    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        callbacks=callbacks,
    )
    
    # Configure DeepSpeed if enabled
    if config.get('parallelism', {}).get('use_deepspeed', False):
        if os.path.exists(config['parallelism']['ds_config_file']):
            trainer.accelerator.state.deepspeed_plugin.deepspeed_config_process(config['parallelism']['ds_config_file'])
    
    # Train model
    logger.info("Starting training...")
    train_result = trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    metrics = train_result.metrics
    
    # Save model
    logger.info(f"Saving model to {output_dir}")
    trainer.save_model(output_dir)
    
    # Save training metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()
    
    # Evaluate if validation data is available
    if validation_dataset is not None:
        logger.info("Running evaluation...")
        eval_metrics = trainer.evaluate()
        
        # Calculate perplexity
        eval_metrics["perplexity"] = math.exp(eval_metrics["eval_loss"])
        
        # Log evaluation metrics
        trainer.log_metrics("eval", eval_metrics)
        trainer.save_metrics("eval", eval_metrics)
        
        # Update metrics with evaluation results
        metrics.update(eval_metrics)
    
    return trainer, metrics 