#!/usr/bin/env python3
"""
Local execution script for model evaluation.
This script provides a simple way to evaluate a trained model locally.
"""

import os
import sys
import argparse
import logging
import json
import time
from pathlib import Path

# Add LiULLM directory to path
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, root_dir)

# Import LiULLM modules
from src.utils.logging import setup_logging
from src.utils.config import load_config
from src.models.llama_model import load_model
from src.training.evaluation import ModelEvaluator, run_comprehensive_evaluation
from src.training.trainer import set_seed

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Evaluate LLM performance locally")
    
    parser.add_argument(
        "--config", 
        type=str, 
        default="configs/evaluation_config.yaml",
        help="Path to evaluation configuration file"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the model to evaluate"
    )
    parser.add_argument(
        "--test_file",
        type=str,
        default=None,
        help="Path to test data file (optional)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/evaluation",
        help="Directory to save evaluation results"
    )
    parser.add_argument(
        "--log_dir",
        type=str,
        default="outputs/logs",
        help="Directory to save logs"
    )
    parser.add_argument(
        "--eval_perplexity",
        action="store_true",
        default=True,
        help="Evaluate perplexity on test set"
    )
    parser.add_argument(
        "--eval_generation",
        action="store_true",
        default=True,
        help="Evaluate text generation"
    )
    parser.add_argument(
        "--eval_instruction",
        action="store_true",
        default=True,
        help="Evaluate instruction following"
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=10,
        help="Number of samples to evaluate for generation and instruction"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode for verbose logging"
    )
    parser.add_argument(
        "--gpu_id",
        type=int,
        default=0,
        help="GPU ID to use for evaluation (single GPU)"
    )
    
    return parser.parse_args()

def setup_directories(directories):
    """Create directories if they don't exist."""
    for directory in directories:
        os.makedirs(directory, exist_ok=True)

def save_evaluation_results(results, output_path, as_json=False):
    """Save evaluation results to a file."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    if as_json:
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
    else:
        with open(output_path, 'w') as f:
            for section, section_results in results.items():
                f.write(f"## {section.upper()}\n\n")
                if isinstance(section_results, dict):
                    for key, value in section_results.items():
                        f.write(f"* {key}: {value}\n")
                else:
                    f.write(f"{section_results}\n")
                f.write("\n")

def main():
    """Main function to run model evaluation."""
    start_time = time.time()
    args = parse_args()
    
    # Setup logging
    log_level = logging.DEBUG if args.debug else logging.INFO
    log_file = os.path.join(args.log_dir, f"evaluation_{int(start_time)}.log")
    setup_logging(log_level=log_level, log_file=log_file)
    logger = logging.getLogger(__name__)
    
    # Create necessary directories
    setup_directories([args.log_dir, args.output_dir])
    
    logger.info("Starting model evaluation")
    logger.info(f"Arguments: {args}")
    
    # Set specific GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    
    # Load configuration
    config = load_config(args.config)
    
    # Update config with CLI arguments if provided
    if args.test_file:
        config['evaluation']['test_file'] = args.test_file
    if args.num_samples:
        config['evaluation']['num_samples'] = args.num_samples
    
    # Set which evaluations to run
    config['evaluation']['eval_perplexity'] = args.eval_perplexity
    config['evaluation']['eval_generation'] = args.eval_generation
    config['evaluation']['eval_instruction'] = args.eval_instruction
    
    # Set seed for reproducibility
    set_seed(config['evaluation'].get('seed', 42))
    
    # Log the configuration
    logger.info(f"Evaluation configuration: {config['evaluation']}")
    
    # Load model and tokenizer
    logger.info(f"Loading model from: {args.model_path}")
    model, tokenizer = load_model(args.model_path)
    
    # Log model information
    logger.info(f"Model loaded with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Evaluate model
    logger.info("Starting model evaluation...")
    
    # Create evaluator
    evaluator = ModelEvaluator(model, tokenizer, config['evaluation'])
    
    # Run comprehensive evaluation
    evaluation_results = run_comprehensive_evaluation(evaluator, config['evaluation'])
    
    # Save results
    timestamp = int(time.time())
    results_file = os.path.join(args.output_dir, f"evaluation_results_{timestamp}.txt")
    results_json = os.path.join(args.output_dir, f"evaluation_results_{timestamp}.json")
    
    save_evaluation_results(evaluation_results, results_file, as_json=False)
    save_evaluation_results(evaluation_results, results_json, as_json=True)
    
    elapsed_time = time.time() - start_time
    logger.info(f"Evaluation completed in {elapsed_time:.2f} seconds")
    logger.info(f"Results saved to: {results_file} and {results_json}")
    
    # Print a summary of the results
    logger.info("Evaluation summary:")
    if 'perplexity' in evaluation_results:
        logger.info(f"- Perplexity: {evaluation_results['perplexity']}")
    if 'generation' in evaluation_results:
        logger.info("- Generation samples evaluated")
    if 'instruction' in evaluation_results:
        logger.info("- Instruction following evaluated")

if __name__ == "__main__":
    main() 