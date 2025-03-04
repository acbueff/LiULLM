"""
Fine-tuning module for instruction tuning of pretrained LLMs.
"""

import os
import json
import logging
import torch
from typing import Dict, Any, Optional, List, Tuple, Union
from datasets import Dataset, load_dataset
from transformers import (
    AutoModelForCausalLM,
    DataCollatorForSeq2Seq,
    PreTrainedTokenizerBase
)
from peft import (
    get_peft_model,
    LoraConfig,
    TaskType,
    PeftModel,
    prepare_model_for_kbit_training
)

from .trainer import train_model, create_training_args

logger = logging.getLogger(__name__)

def prepare_instruction_dataset(
    tokenizer: PreTrainedTokenizerBase,
    train_file: str,
    validation_file: Optional[str] = None,
    max_seq_length: int = 2048,
    prompt_template: str = "<s>User: {instruction}\nAssistant: ",
    completion_template: str = "{response}</s>",
    preprocessing_num_workers: Optional[int] = None,
    overwrite_cache: bool = False
) -> Tuple[Dataset, Optional[Dataset]]:
    """
    Prepare instruction datasets for fine-tuning.
    
    Args:
        tokenizer: Tokenizer to use for encoding.
        train_file: Path to training data file (JSONL with instruction/response pairs).
        validation_file: Path to validation data file.
        max_seq_length: Maximum sequence length for training.
        prompt_template: Template for formatting instructions (with {instruction} placeholder).
        completion_template: Template for formatting responses (with {response} placeholder).
        preprocessing_num_workers: Number of workers for preprocessing.
        overwrite_cache: Whether to overwrite cached datasets.
        
    Returns:
        Tuple of (train_dataset, val_dataset).
    """
    logger.info(f"Loading instruction datasets from {train_file}")
    
    data_files = {"train": train_file}
    if validation_file:
        data_files["validation"] = validation_file
    
    # Load raw datasets
    extension = train_file.split(".")[-1]
    raw_datasets = load_dataset(
        extension,
        data_files=data_files,
        cache_dir=None
    )
    
    # Function to format and tokenize examples
    def process_example(example):
        # Create full prompt with instruction
        if "instruction" in example:
            instruction = example["instruction"]
        elif "prompt" in example:
            instruction = example["prompt"]
        else:
            instruction = example.get("input", "")
            
        # Get response/completion
        if "response" in example:
            response = example["response"]
        elif "completion" in example:
            response = example["completion"]
        else:
            response = example.get("output", "")
            
        # Format with templates
        prompt = prompt_template.format(instruction=instruction)
        completion = completion_template.format(response=response)
        full_text = prompt + completion
        
        # Tokenize
        tokenized = tokenizer(
            full_text,
            padding="max_length",
            truncation=True,
            max_length=max_seq_length,
            return_tensors=None
        )
        
        # Find the input_ids corresponding to the completion start
        prompt_ids = tokenizer(
            prompt,
            padding=False,
            truncation=False,
            return_tensors=None
        )["input_ids"]
        prompt_length = len(prompt_ids)
        
        # Create labels: -100 for prompt tokens (ignored in loss), actual ids for completion
        labels = [-100] * prompt_length + tokenized["input_ids"][prompt_length:]
        
        # Ensure labels match input_ids length
        if len(labels) < len(tokenized["input_ids"]):
            labels += [-100] * (len(tokenized["input_ids"]) - len(labels))
        elif len(labels) > len(tokenized["input_ids"]):
            labels = labels[:len(tokenized["input_ids"])]
            
        tokenized["labels"] = labels
        return tokenized
    
    # Process datasets
    processed_datasets = raw_datasets.map(
        process_example,
        remove_columns=raw_datasets["train"].column_names,
        num_proc=preprocessing_num_workers,
        load_from_cache_file=not overwrite_cache,
        desc="Processing instruction datasets",
    )
    
    train_dataset = processed_datasets["train"]
    validation_dataset = processed_datasets["validation"] if "validation" in processed_datasets else None
    
    logger.info(f"Train dataset size: {len(train_dataset)}")
    if validation_dataset:
        logger.info(f"Validation dataset size: {len(validation_dataset)}")
        
    return train_dataset, validation_dataset

def setup_peft(
    model: AutoModelForCausalLM,
    config: Dict[str, Any]
) -> PeftModel:
    """
    Set up Parameter-Efficient Fine-Tuning (PEFT) for the model.
    
    Args:
        model: Model to fine-tune.
        config: Fine-tuning configuration.
        
    Returns:
        Model wrapped with PEFT.
    """
    peft_config = config.get("peft", {})
    if not peft_config.get("use_peft", False):
        logger.info("PEFT is disabled, using full model fine-tuning")
        return model
    
    peft_method = peft_config.get("method", "lora").lower()
    
    if peft_method == "lora":
        logger.info("Setting up LoRA fine-tuning")
        
        # Get LoRA parameters
        lora_r = peft_config.get("lora_r", 8)
        lora_alpha = peft_config.get("lora_alpha", 16)
        lora_dropout = peft_config.get("lora_dropout", 0.05)
        
        # Get target modules
        target_modules = peft_config.get("target_modules", ["q_proj", "k_proj", "v_proj", "o_proj"])
        
        # Configure LoRA
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=target_modules,
            bias="none",
        )
        
        # Prepare model if it's quantized
        if getattr(model, "is_quantized", False):
            logger.info("Preparing quantized model for PEFT")
            model = prepare_model_for_kbit_training(
                model, 
                use_gradient_checkpointing=config.get("model", {}).get("gradient_checkpointing", False)
            )
        
        # Wrap model with LoRA
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
    else:
        logger.warning(f"Unsupported PEFT method: {peft_method}, using full fine-tuning")
        
    return model

def finetune_model(
    model,
    tokenizer,
    train_dataset: Dataset,
    validation_dataset: Optional[Dataset],
    config: Dict[str, Any],
    output_dir: str,
    resume_from_checkpoint: Optional[Union[str, bool]] = None
) -> Tuple[Any, Dict[str, float]]:
    """
    Fine-tune a pretrained model on instruction data.
    
    Args:
        model: Model to fine-tune.
        tokenizer: Tokenizer for the model.
        train_dataset: Training dataset.
        validation_dataset: Validation dataset.
        config: Configuration dictionary.
        output_dir: Output directory.
        resume_from_checkpoint: Whether to resume from checkpoint.
        
    Returns:
        Tuple of (model, metrics).
    """
    logger.info("Setting up instruction fine-tuning...")
    
    # Set up PEFT if enabled
    if config.get("peft", {}).get("use_peft", False):
        model = setup_peft(model, config)
    
    # Configure data collator
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        pad_to_multiple_of=8,
        return_tensors="pt",
        padding=True,
    )
    
    # Create trainer
    trainer, metrics = train_model(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        validation_dataset=validation_dataset,
        config=config,
        output_dir=output_dir,
        resume_from_checkpoint=resume_from_checkpoint
    )
    
    # Save full model
    logger.info(f"Saving fine-tuned model to {output_dir}")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    # Save fine-tuning config
    with open(os.path.join(output_dir, "finetune_config.json"), "w") as f:
        json.dump(config, f, indent=2)
    
    return model, metrics 