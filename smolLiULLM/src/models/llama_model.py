"""
LLaMA model initialization and configuration module.
"""

import os
import torch
import logging
from typing import Dict, Any, Optional, Union
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    PreTrainedTokenizerBase,
    LlamaConfig,
    LlamaForCausalLM
)

logger = logging.getLogger(__name__)

def create_model_config(config: Dict[str, Any]) -> LlamaConfig:
    """
    Create a LLaMA model configuration from a configuration dictionary.
    
    Args:
        config: Model configuration dictionary.
        
    Returns:
        LlamaConfig object.
    """
    model_config = config['model']
    
    # Check if we're using a base model as reference
    if model_config.get('base_model'):
        logger.info(f"Using base model configuration from: {model_config['base_model']}")
        base_config = AutoConfig.from_pretrained(model_config['base_model'])
        
        # Override parameters from our config
        config_dict = base_config.to_dict()
        for key, value in model_config.items():
            if key != 'base_model' and key in config_dict:
                config_dict[key] = value
                
        # Ensure vocab size is properly set
        if 'vocab_size' in model_config:
            config_dict['vocab_size'] = model_config['vocab_size']
            
        llama_config = LlamaConfig.from_dict(config_dict)
    else:
        # Create a config from scratch
        logger.info("Creating a new LLaMA model configuration from scratch")
        
        llama_config = LlamaConfig(
            vocab_size=model_config.get('vocab_size', 32000),
            hidden_size=model_config.get('hidden_size', 2048),
            intermediate_size=model_config.get('intermediate_size', 5632),  # ~2.75 * hidden_size
            num_hidden_layers=model_config.get('num_hidden_layers', 16),
            num_attention_heads=model_config.get('num_attention_heads', 16),
            max_position_embeddings=model_config.get('max_position_embeddings', 2048),
            rms_norm_eps=model_config.get('rms_norm_eps', 1e-6),
            initializer_range=model_config.get('initializer_range', 0.02),
            use_cache=model_config.get('use_cache', True),
            pad_token_id=None,  # LLaMA doesn't have a pad token by default
            bos_token_id=model_config.get('bos_token_id', 1),
            eos_token_id=model_config.get('eos_token_id', 2),
            tie_word_embeddings=model_config.get('tie_word_embeddings', False),
            architectures=["LlamaForCausalLM"],
        )
    
    # Add model initialization parameters
    if 'initialization' in config:
        init_config = config['initialization']
        llama_config.initializer_range = init_config.get('init_std', 0.02)
        
    # Enable gradient checkpointing if specified
    if model_config.get('gradient_checkpointing', False):
        llama_config.use_cache = False  # Required for gradient checkpointing
        
    logger.info(f"Created model config with {llama_config.num_hidden_layers} layers, "
               f"{llama_config.hidden_size} hidden size, {llama_config.vocab_size} vocab size")
    
    return llama_config

def create_model(
    config: Dict[str, Any],
    pretrained_model_path: Optional[str] = None,
    local_rank: int = -1
) -> LlamaForCausalLM:
    """
    Create a LLaMA model based on the configuration.
    
    Args:
        config: Model configuration dictionary.
        pretrained_model_path: Path to pretrained model to load. If None, initialize from scratch.
        local_rank: Local rank for distributed training.
        
    Returns:
        LlamaForCausalLM model instance.
    """
    model_config = config['model']
    
    # Set device-aware settings
    torch_dtype = torch.float16 if config.get('mixed_precision', {}).get('enabled', False) else torch.float32
    device_map = "auto" if local_rank == -1 else {"": local_rank}
    
    # Load pretrained model if specified
    if pretrained_model_path or model_config.get('use_pretrained', False):
        # Path to the model checkpoint or HF model ID
        model_path = pretrained_model_path or model_config['base_model']
        logger.info(f"Loading pretrained model from: {model_path}")
        
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch_dtype,
            device_map=device_map,
            trust_remote_code=False
        )
    else:
        # Initialize a new model from scratch
        logger.info("Initializing a new LLaMA model from scratch")
        
        # Create model configuration
        llama_config = create_model_config(config)
        
        # Create model
        model = LlamaForCausalLM(config=llama_config)
        
        # Initialize with custom seed if specified
        if 'initialization' in config and 'random_seed' in config['initialization']:
            seed = config['initialization']['random_seed']
            logger.info(f"Setting random seed to {seed} for model initialization")
            torch.manual_seed(seed)
            
    # Enable gradient checkpointing if specified (for memory efficiency)
    if model_config.get('gradient_checkpointing', False):
        logger.info("Enabling gradient checkpointing for memory efficiency")
        model.gradient_checkpointing_enable()
    
    return model

def save_model(
    model: LlamaForCausalLM,
    tokenizer: PreTrainedTokenizerBase,
    output_dir: str,
    save_full_model: bool = True
) -> None:
    """
    Save a trained model and tokenizer.
    
    Args:
        model: Trained model instance.
        tokenizer: Tokenizer instance.
        output_dir: Directory to save the model to.
        save_full_model: Whether to save the full model (True) or just the model weights (False).
    """
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"Saving model to {output_dir}")
    
    # Save model
    if save_full_model:
        model.save_pretrained(output_dir)
    else:
        # Save just the weights
        torch.save(model.state_dict(), os.path.join(output_dir, "pytorch_model.bin"))
    
    # Save tokenizer
    tokenizer.save_pretrained(output_dir)
    
    logger.info(f"Model and tokenizer saved to {output_dir}")
    
def load_model(
    model_dir: str,
    tokenizer_dir: Optional[str] = None,
    local_rank: int = -1,
    torch_dtype: Optional[torch.dtype] = None
) -> tuple:
    """
    Load a saved model and tokenizer.
    
    Args:
        model_dir: Directory with the saved model.
        tokenizer_dir: Directory with the saved tokenizer. If None, uses model_dir.
        local_rank: Local rank for distributed training.
        torch_dtype: Data type for model loading (fp16, bf16, or fp32).
        
    Returns:
        Tuple of (model, tokenizer).
    """
    logger.info(f"Loading model from {model_dir}")
    
    # Set device mapping
    device_map = "auto" if local_rank == -1 else {"": local_rank}
    
    # Set dtype if not specified
    if torch_dtype is None:
        if torch.cuda.is_available():
            torch_dtype = torch.float16
        else:
            torch_dtype = torch.float32
    
    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        torch_dtype=torch_dtype,
        device_map=device_map,
        trust_remote_code=False
    )
    
    # Load tokenizer
    if tokenizer_dir is None:
        tokenizer_dir = model_dir
        
    tokenizer = AutoModelForCausalLM.from_pretrained(tokenizer_dir)
    
    logger.info(f"Loaded model with {model.config.num_hidden_layers} layers, "
               f"{model.config.hidden_size} hidden size")
    
    return model, tokenizer 