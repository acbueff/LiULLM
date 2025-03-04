"""
Logging utilities for consistent logging across the pipeline.
"""

import os
import sys
import logging
import json
from typing import Optional, Dict, Any, Union
from datetime import datetime

def setup_logging(
    log_dir: str = "outputs/logs",
    log_level: int = logging.INFO,
    log_to_console: bool = True,
    log_to_file: bool = True,
    experiment_name: Optional[str] = None,
) -> logging.Logger:
    """
    Set up logging to file and console.
    
    Args:
        log_dir: Directory to store log files.
        log_level: Logging level (e.g., logging.INFO, logging.DEBUG).
        log_to_console: Whether to log to console.
        log_to_file: Whether to log to file.
        experiment_name: Name of the experiment for the log file name.
        
    Returns:
        Configured root logger.
    """
    # Create logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    
    # Remove all existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Set up console handler
    if log_to_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        console_handler.setFormatter(console_formatter)
        console_handler.setLevel(log_level)
        root_logger.addHandler(console_handler)
    
    # Set up file handler
    if log_to_file:
        os.makedirs(log_dir, exist_ok=True)
        
        if experiment_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            experiment_name = f"run_{timestamp}"
            
        log_file = os.path.join(log_dir, f"{experiment_name}.log")
        file_handler = logging.FileHandler(log_file, mode='a')
        file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(file_formatter)
        file_handler.setLevel(log_level)
        root_logger.addHandler(file_handler)
        
        logging.info(f"Logging to {log_file}")
    
    return root_logger

def log_config(config: Dict[str, Any], logger: Optional[logging.Logger] = None) -> None:
    """
    Log configuration parameters.
    
    Args:
        config: Configuration dictionary to log.
        logger: Logger to use. If None, uses the root logger.
    """
    if logger is None:
        logger = logging.getLogger()
    
    logger.info("Configuration:")
    config_str = json.dumps(config, indent=2)
    # Split by lines to have each config parameter on its own log line
    for line in config_str.split('\n'):
        logger.info(line)

def get_experiment_name(
    prefix: str = "",
    config: Optional[Dict[str, Any]] = None,
    include_timestamp: bool = True
) -> str:
    """
    Generate a unique experiment name based on configuration.
    
    Args:
        prefix: Prefix for the experiment name.
        config: Configuration dictionary to extract parameters from.
        include_timestamp: Whether to include timestamp in the name.
        
    Returns:
        Experiment name string.
    """
    parts = []
    
    if prefix:
        parts.append(prefix)
    
    if config is not None:
        # Add key configuration parameters to name
        if 'model' in config and 'num_hidden_layers' in config['model']:
            parts.append(f"L{config['model']['num_hidden_layers']}")
        
        if 'model' in config and 'hidden_size' in config['model']:
            parts.append(f"H{config['model']['hidden_size']}")
        
        if 'training' in config and 'learning_rate' in config['training']:
            # Format learning rate (e.g., 2e-4 -> 2e-4)
            lr = config['training']['learning_rate']
            parts.append(f"lr{lr:.0e}" if isinstance(lr, float) else f"lr{lr}")
    
    if include_timestamp:
        timestamp = datetime.now().strftime("%m%d_%H%M")
        parts.append(timestamp)
    
    return "-".join(parts)

def log_metrics(metrics: Dict[str, Union[float, int]], step: int, logger: Optional[logging.Logger] = None) -> None:
    """
    Log metrics for a specific step.
    
    Args:
        metrics: Dictionary of metric name to value.
        step: Current training step.
        logger: Logger to use. If None, uses the root logger.
    """
    if logger is None:
        logger = logging.getLogger()
    
    log_str = f"Step {step} metrics: "
    log_str += ", ".join([f"{k}={v:.6f}" if isinstance(v, float) else f"{k}={v}" for k, v in metrics.items()])
    logger.info(log_str) 