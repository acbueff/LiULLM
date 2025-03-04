"""
Evaluation module for model evaluation and benchmarking.
"""

import os
import json
import logging
import math
import torch
import numpy as np
from typing import Dict, Any, List, Optional, Tuple, Union
from datasets import Dataset, load_dataset
from transformers import (
    PreTrainedModel,
    PreTrainedTokenizerBase,
    AutoModelForCausalLM,
    AutoTokenizer,
    TextGenerationPipeline
)
import evaluate

logger = logging.getLogger(__name__)

class ModelEvaluator:
    """
    Class for evaluating language models.
    """
    
    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        device: Optional[Union[str, torch.device]] = None,
        max_length: int = 2048,
        batch_size: int = 1
    ):
        """
        Initialize model evaluator.
        
        Args:
            model: Model to evaluate.
            tokenizer: Tokenizer for the model.
            device: Device to run evaluation on.
            max_length: Maximum sequence length for generation.
            batch_size: Batch size for evaluation.
        """
        self.model = model
        self.tokenizer = tokenizer
        
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        
        self.max_length = max_length
        self.batch_size = batch_size
        
        # Move model to device
        self.model.to(self.device)
        self.model.eval()
        
        # Set up generation pipeline
        self.pipeline = TextGenerationPipeline(
            model=model,
            tokenizer=tokenizer,
            device=0 if device == "cuda" else -1,
            max_length=max_length,
            batch_size=batch_size
        )
        
    def compute_perplexity(
        self,
        texts: Union[str, List[str]],
        stride: int = 512,
    ) -> Dict[str, float]:
        """
        Compute perplexity on texts.
        
        Args:
            texts: Text or list of texts to evaluate.
            stride: Stride for sliding window to handle long texts.
            
        Returns:
            Dictionary with perplexity metrics.
        """
        logger.info("Computing perplexity...")
        
        if isinstance(texts, str):
            texts = [texts]
            
        # Tokenize and encode texts
        encodings = self.tokenizer(texts, return_tensors="pt", padding=True)
        
        # Move tensors to device
        input_ids = encodings.input_ids.to(self.device)
        attention_mask = encodings.attention_mask.to(self.device)
        
        total_loss = 0.0
        total_tokens = 0
        
        # Processing in batches with sliding window if needed
        for i in range(0, input_ids.shape[0], self.batch_size):
            batch_input_ids = input_ids[i:i+self.batch_size]
            batch_attention_mask = attention_mask[i:i+self.batch_size]
            
            # Using sliding window for long texts
            seq_len = batch_input_ids.size(1)
            
            # Initialize tensor to store losses
            nlls = torch.zeros(batch_input_ids.size(0), device=self.device)
            prev_end_loc = 0
            
            for begin_loc in range(0, seq_len, stride):
                end_loc = min(begin_loc + self.max_length, seq_len)
                target_len = end_loc - prev_end_loc
                
                # Select current window
                input_ids_chunk = batch_input_ids[:, begin_loc:end_loc]
                target_ids_chunk = input_ids_chunk.clone()
                
                # Set first tokens to -100 to ignore in loss calculation
                if begin_loc > 0:
                    target_ids_chunk[:, :stride] = -100
                
                with torch.no_grad():
                    outputs = self.model(input_ids_chunk, labels=target_ids_chunk)
                    loss = outputs.loss
                
                # Accumulate loss
                nlls += loss.detach() * target_len
                prev_end_loc = end_loc
                
                if end_loc == seq_len:
                    break
                    
            # Average per-token loss
            batch_token_count = batch_attention_mask.sum(dim=1)
            batch_loss = nlls / batch_token_count
            
            total_loss += batch_loss.sum().item()
            total_tokens += batch_input_ids.shape[0]
            
        # Calculate average perplexity across all texts
        avg_loss = total_loss / total_tokens
        perplexity = math.exp(avg_loss)
        
        return {"perplexity": perplexity, "loss": avg_loss}
    
    def evaluate_language_perplexity(
        self,
        data_files: Dict[str, str]
    ) -> Dict[str, Dict[str, float]]:
        """
        Evaluate perplexity on texts by language.
        
        Args:
            data_files: Dict mapping language names to validation file paths.
            
        Returns:
            Dictionary mapping language names to perplexity metrics.
        """
        results = {}
        
        for lang, file_path in data_files.items():
            logger.info(f"Evaluating perplexity for language: {lang}")
            
            try:
                # Load data for this language
                with open(file_path, "r", encoding="utf-8") as f:
                    texts = [line.strip() for line in f if line.strip()]
                
                # Compute perplexity
                metrics = self.compute_perplexity(texts)
                results[lang] = metrics
                
                logger.info(f"{lang} perplexity: {metrics['perplexity']:.2f}")
                
            except Exception as e:
                logger.error(f"Error evaluating {lang}: {e}")
                results[lang] = {"perplexity": float("inf"), "error": str(e)}
        
        # Compute average perplexity across all languages
        if results:
            avg_perplexity = np.mean([res["perplexity"] for lang, res in results.items() 
                                    if not math.isinf(res["perplexity"])])
            results["average"] = {"perplexity": avg_perplexity}
            logger.info(f"Average perplexity across languages: {avg_perplexity:.2f}")
        
        return results
    
    def generate_text(
        self,
        prompts: Union[str, List[str]],
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        repetition_penalty: float = 1.2,
        do_sample: bool = True,
        **kwargs
    ) -> List[str]:
        """
        Generate text completions for prompts.
        
        Args:
            prompts: Prompt or list of prompts.
            max_new_tokens: Maximum number of new tokens to generate.
            temperature: Sampling temperature.
            top_p: Nucleus sampling probability.
            repetition_penalty: Penalty for repeating tokens.
            do_sample: Whether to use sampling or greedy generation.
            **kwargs: Additional arguments for generation.
            
        Returns:
            List of generated texts.
        """
        if isinstance(prompts, str):
            prompts = [prompts]
            
        logger.info(f"Generating completions for {len(prompts)} prompts")
        
        generation_kwargs = {
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "repetition_penalty": repetition_penalty,
            "do_sample": do_sample,
            **kwargs
        }
        
        outputs = self.pipeline(
            prompts,
            **generation_kwargs
        )
        
        # Extract generated texts
        results = []
        for out in outputs:
            if isinstance(out, list):
                # If multiple sequences per prompt
                results.append(out[0]["generated_text"])
            else:
                results.append(out["generated_text"])
        
        return results
    
    def evaluate_instruction_following(
        self,
        instruction_file: str,
        output_dir: str = "outputs/evaluation/instruction_results",
        prompt_template: str = "<s>User: {instruction}\nAssistant: ",
        **kwargs
    ) -> Dict[str, Any]:
        """
        Evaluate model on instruction-following examples.
        
        Args:
            instruction_file: Path to instruction examples JSON file.
            output_dir: Directory to save generation results.
            prompt_template: Template for formatting instructions.
            **kwargs: Additional arguments for text generation.
            
        Returns:
            Dictionary with evaluation metrics.
        """
        logger.info(f"Evaluating instruction-following on {instruction_file}")
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Load instruction data
        with open(instruction_file, "r", encoding="utf-8") as f:
            instructions = [json.loads(line) for line in f]
            
        prompts = []
        references = []
        
        for item in instructions:
            if "instruction" in item:
                instruction = item["instruction"]
            elif "prompt" in item:
                instruction = item["prompt"]
            else:
                instruction = item.get("input", "")
                
            # Get reference response if available
            if "response" in item:
                reference = item["response"]
            elif "completion" in item:
                reference = item["completion"]
            else:
                reference = item.get("output", "")
                
            # Format prompt
            prompt = prompt_template.format(instruction=instruction)
            
            prompts.append(prompt)
            references.append(reference)
        
        # Generate completions
        generated_texts = self.generate_text(prompts, **kwargs)
        
        # Process generated texts to extract only the response part
        processed_responses = []
        for text in generated_texts:
            # Attempt to extract just the model's response by removing the prompt
            for prompt in prompts:
                if text.startswith(prompt):
                    response = text[len(prompt):]
                    processed_responses.append(response)
                    break
            else:
                # If prompt not found, use the full text
                processed_responses.append(text)
        
        # Calculate BLEU and other metrics if references are available
        metrics = {}
        if any(references):
            bleu = evaluate.load("bleu")
            rouge = evaluate.load("rouge")
            
            # Calculate BLEU
            bleu_results = bleu.compute(predictions=processed_responses, references=references)
            metrics["bleu"] = bleu_results["bleu"]
            
            # Calculate ROUGE
            rouge_results = rouge.compute(predictions=processed_responses, references=references)
            metrics.update(rouge_results)
            
            logger.info(f"BLEU score: {metrics['bleu']:.4f}")
            logger.info(f"ROUGE-L score: {metrics['rougeL']:.4f}")
        
        # Save results for manual inspection
        results = []
        for i, (prompt, ref, gen) in enumerate(zip(prompts, references, processed_responses)):
            results.append({
                "id": i,
                "prompt": prompt,
                "reference": ref,
                "generated": gen
            })
            
        with open(os.path.join(output_dir, "generation_results.jsonl"), "w", encoding="utf-8") as f:
            for item in results:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
                
        # Save metrics
        with open(os.path.join(output_dir, "metrics.json"), "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
            
        return metrics
    
    def run_comprehensive_evaluation(
        self,
        eval_config: Dict[str, Any],
        output_dir: str = "outputs/evaluation"
    ) -> Dict[str, Dict[str, Any]]:
        """
        Run a comprehensive evaluation of the model.
        
        Args:
            eval_config: Evaluation configuration.
            output_dir: Directory to save evaluation results.
            
        Returns:
            Dictionary with evaluation results.
        """
        logger.info("Running comprehensive model evaluation")
        
        os.makedirs(output_dir, exist_ok=True)
        results = {}
        
        # 1. Evaluate perplexity by language
        if "perplexity" in eval_config and "language_files" in eval_config["perplexity"]:
            language_files = eval_config["perplexity"]["language_files"]
            logger.info(f"Evaluating perplexity on {len(language_files)} languages")
            
            perplexity_results = self.evaluate_language_perplexity(language_files)
            results["perplexity"] = perplexity_results
            
            # Save perplexity results
            with open(os.path.join(output_dir, "perplexity_results.json"), "w") as f:
                json.dump(perplexity_results, f, indent=2)
        
        # 2. Evaluate instruction following if configured
        if "instruction" in eval_config and "file" in eval_config["instruction"]:
            instruction_file = eval_config["instruction"]["file"]
            prompt_template = eval_config["instruction"].get(
                "prompt_template", "<s>User: {instruction}\nAssistant: "
            )
            
            logger.info(f"Evaluating instruction following on {instruction_file}")
            
            instruction_results = self.evaluate_instruction_following(
                instruction_file=instruction_file,
                output_dir=os.path.join(output_dir, "instruction_results"),
                prompt_template=prompt_template,
                max_new_tokens=eval_config["instruction"].get("max_new_tokens", 512),
                temperature=eval_config["instruction"].get("temperature", 0.7),
                top_p=eval_config["instruction"].get("top_p", 0.9),
                repetition_penalty=eval_config["instruction"].get("repetition_penalty", 1.2)
            )
            
            results["instruction"] = instruction_results
        
        # 3. Save all evaluation results
        with open(os.path.join(output_dir, "evaluation_results.json"), "w") as f:
            json.dump(results, f, indent=2)
            
        logger.info("Comprehensive evaluation complete")
        return results


# Load and create evaluator
def create_evaluator(
    model_path: str,
    tokenizer_path: Optional[str] = None,
    device: Optional[str] = None,
    **kwargs
) -> ModelEvaluator:
    """
    Load model and create an evaluator.
    
    Args:
        model_path: Path to the model.
        tokenizer_path: Path to the tokenizer (defaults to model_path).
        device: Device to run evaluation on.
        **kwargs: Additional arguments for ModelEvaluator.
        
    Returns:
        ModelEvaluator instance.
    """
    logger.info(f"Loading model from {model_path}")
    
    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if device is None else None
    )
    
    # Load tokenizer
    if tokenizer_path is None:
        tokenizer_path = model_path
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    
    # Create evaluator
    evaluator = ModelEvaluator(
        model=model,
        tokenizer=tokenizer,
        device=device,
        **kwargs
    )
    
    return evaluator 