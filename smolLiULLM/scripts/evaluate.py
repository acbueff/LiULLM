#!/usr/bin/env python3
"""
Evaluation script for measuring model performance on perplexity,
text generation, and instruction following metrics.
"""

import os
import sys
import argparse
import logging
import json
from typing import Dict, Any, List

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ""))

from src.utils.config import load_config
from src.utils.logging import setup_logging, log_config
from src.models.llama_model import load_model
from src.training.evaluation import ModelEvaluator, run_comprehensive_evaluation
from src.training.trainer import set_seed

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Evaluate language model performance")
    
    parser.add_argument(
        "--config", 
        type=str, 
        default="configs/evaluation_config.yaml",
        help="Path to evaluation configuration file"
    )
    parser.add_argument(
        "--model_path", 
        type=str, 
        default=None,
        help="Path to model (overrides config)"
    )
    parser.add_argument(
        "--tokenizer_path", 
        type=str, 
        default=None,
        help="Path to tokenizer (overrides config)"
    )
    parser.add_argument(
        "--eval_data_dir", 
        type=str, 
        default=None,
        help="Directory with evaluation data (overrides config)"
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default=None,
        help="Directory to save evaluation results (overrides config)"
    )
    parser.add_argument(
        "--mode", 
        type=str, 
        choices=["perplexity", "generation", "instruction", "all"],
        default="all",
        help="Evaluation mode (perplexity, generation, instruction, or all)"
    )
    parser.add_argument(
        "--max_samples", 
        type=int, 
        default=None,
        help="Maximum number of samples to evaluate (overrides config)"
    )
    parser.add_argument(
        "--batch_size", 
        type=int, 
        default=None,
        help="Batch size for evaluation (overrides config)"
    )
    parser.add_argument(
        "--language_subset", 
        type=str, 
        default=None,
        help="Comma-separated list of language codes to evaluate (e.g., 'en,sv,de')"
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
        "--json_output", 
        action="store_true",
        help="Output detailed results in JSON format"
    )
    parser.add_argument(
        "--debug", 
        action="store_true",
        help="Enable debug mode with more verbose logging"
    )
    
    return parser.parse_args()

def save_evaluation_results(results: Dict[str, Any], output_path: str, as_json: bool = False):
    """Save evaluation results to file."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    if as_json:
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
    else:
        with open(output_path, 'w') as f:
            for section, section_results in results.items():
                f.write(f"=== {section} ===\n")
                if isinstance(section_results, dict):
                    for k, v in section_results.items():
                        f.write(f"{k}: {v}\n")
                else:
                    f.write(f"{section_results}\n")
                f.write("\n")

def main():
    """Main function to run evaluation."""
    # Parse arguments
    args = parse_args()
    
    # Set up logging
    log_level = logging.DEBUG if args.debug else logging.INFO
    logger = setup_logging(
        log_dir=args.log_dir,
        log_level=log_level,
        experiment_name="evaluation"
    )
    
    try:
        # Load configuration
        logger.info(f"Loading evaluation configuration from {args.config}")
        config = load_config(args.config)
        
        # Override config with CLI arguments if provided
        if args.model_path:
            config['model_path'] = args.model_path
            logger.info(f"Using model from: {args.model_path}")
        
        if args.tokenizer_path:
            config['tokenizer_path'] = args.tokenizer_path
            logger.info(f"Using tokenizer from: {args.tokenizer_path}")
        
        if args.eval_data_dir:
            config['eval_data_dir'] = args.eval_data_dir
            logger.info(f"Using evaluation data from: {args.eval_data_dir}")
        
        if args.output_dir:
            config['output_dir'] = args.output_dir
            os.makedirs(args.output_dir, exist_ok=True)
            logger.info(f"Using output directory: {args.output_dir}")
        
        if args.max_samples:
            config['max_samples'] = args.max_samples
            logger.info(f"Using max samples: {args.max_samples}")
        
        if args.batch_size:
            config['batch_size'] = args.batch_size
            logger.info(f"Using batch size: {args.batch_size}")
        
        if args.language_subset:
            language_list = args.language_subset.split(',')
            config['language_subset'] = language_list
            logger.info(f"Evaluating on languages: {language_list}")
        
        if args.seed is not None:
            config['seed'] = args.seed
            logger.info(f"Using random seed: {args.seed}")
        
        # Set evaluation mode
        if args.mode != "all":
            logger.info(f"Evaluation mode: {args.mode}")
            config['evaluation_modes'] = [args.mode]
        else:
            logger.info("Performing comprehensive evaluation (all modes)")
        
        # Log configuration
        log_config(config)
        
        # Set random seed for reproducibility
        set_seed(config.get('seed', 42))
        
        # Load model
        logger.info(f"Loading model from {config['model_path']}")
        model, tokenizer = load_model(
            model_dir=config['model_path'],
            tokenizer_path=config['tokenizer_path']
        )
        
        # Create evaluator
        logger.info("Initializing model evaluator")
        evaluator = ModelEvaluator(
            model=model,
            tokenizer=tokenizer,
            config=config
        )
        
        # Run evaluation
        logger.info("Starting evaluation...")
        results = run_comprehensive_evaluation(evaluator, config)
        
        # Save results
        result_path = os.path.join(config['output_dir'], "evaluation_results")
        if args.json_output:
            result_path += ".json"
            logger.info(f"Saving detailed evaluation results to {result_path}")
            save_evaluation_results(results, result_path, as_json=True)
        else:
            result_path += ".txt"
            logger.info(f"Saving evaluation results to {result_path}")
            save_evaluation_results(results, result_path, as_json=False)
        
        # Log summary results
        logger.info("Evaluation completed. Summary:")
        if 'perplexity' in results:
            logger.info(f"Average Perplexity: {results['perplexity'].get('avg_perplexity', 'N/A')}")
        
        if 'generation' in results:
            logger.info(f"Generation Quality: {results['generation'].get('avg_quality_score', 'N/A')}")
        
        if 'instruction' in results:
            logger.info(f"Instruction BLEU: {results['instruction'].get('avg_bleu', 'N/A')}")
            logger.info(f"Instruction ROUGE-L: {results['instruction'].get('avg_rouge_l', 'N/A')}")
        
        logger.info(f"All evaluation results saved to {result_path}")
        
    except Exception as e:
        logger.error(f"Error in evaluation: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main() 