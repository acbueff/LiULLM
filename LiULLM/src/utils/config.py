"""
Configuration utilities for loading and managing YAML config files.
"""

import os
import yaml
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from a YAML file.
    
    Args:
        config_path: Path to the YAML config file.
        
    Returns:
        Dictionary containing the configuration.
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        logger.info(f"Loaded configuration from {config_path}")
        return config
    except Exception as e:
        logger.error(f"Error loading config from {config_path}: {e}")
        raise

def save_config(config: Dict[str, Any], output_path: str) -> None:
    """
    Save a configuration dictionary to a YAML file.
    
    Args:
        config: Configuration dictionary to save.
        output_path: Path where to save the YAML file.
    """
    try:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            yaml.dump(config, f, default_flow_style=False)
        logger.info(f"Saved configuration to {output_path}")
    except Exception as e:
        logger.error(f"Error saving config to {output_path}: {e}")
        raise

def update_config(config: Dict[str, Any], updates: Dict[str, Any]) -> Dict[str, Any]:
    """
    Update a configuration with new values.
    
    Args:
        config: Original configuration dictionary.
        updates: Dictionary of updates to apply.
        
    Returns:
        Updated configuration dictionary.
    """
    # Deep update the config with the updates
    def _update_dict(d, u):
        for k, v in u.items():
            if isinstance(v, dict) and k in d and isinstance(d[k], dict):
                d[k] = _update_dict(d[k], v)
            else:
                d[k] = v
        return d
    
    updated_config = _update_dict(config.copy(), updates)
    return updated_config

def merge_configs(*configs: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merge multiple configuration dictionaries, with later ones taking precedence.
    
    Args:
        *configs: Configuration dictionaries to merge.
        
    Returns:
        Merged configuration dictionary.
    """
    result = {}
    for config in configs:
        result = update_config(result, config)
    return result

def get_config_with_cli_overrides(config_path: str, cli_args: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Load a config file and override values with command line arguments.
    
    Args:
        config_path: Path to the YAML config file.
        cli_args: Dictionary of command line arguments to override config values.
        
    Returns:
        Final configuration dictionary.
    """
    config = load_config(config_path)
    
    if cli_args:
        # Convert flat CLI args to nested structure if needed
        updates = {}
        for key, value in cli_args.items():
            if '.' in key:
                # Handle nested keys like 'model.learning_rate'
                parts = key.split('.')
                current = updates
                for part in parts[:-1]:
                    if part not in current:
                        current[part] = {}
                    current = current[part]
                current[parts[-1]] = value
            else:
                updates[key] = value
        
        config = update_config(config, updates)
    
    return config 