"""
Tokenizer training module.
"""

import os
import logging
from typing import Dict, Any, List, Optional
from tokenizers import Tokenizer, models, pre_tokenizers, decoders, processors, trainers
from tokenizers.models import BPE, Unigram, WordPiece, WordLevel
from tokenizers.pre_tokenizers import Whitespace, ByteLevel
from tokenizers.trainers import BpeTrainer, UnigramTrainer, WordPieceTrainer, WordLevelTrainer
from transformers import PreTrainedTokenizerFast

logger = logging.getLogger(__name__)

class TokenizerTrainer:
    """Class for training a tokenizer on multilingual text data."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the tokenizer trainer.
        
        Args:
            config: Tokenizer configuration dictionary.
        """
        self.config = config
        self.model_type = config.get('model_type', 'bpe').lower()
        self.byte_level = config.get('byte_level', True)
        self.vocab_size = config.get('vocab_size', 32000)
        self.min_frequency = config.get('min_frequency', 2)
        self.special_tokens = config.get('special_tokens', [])
        self.output_dir = config.get('output_dir', 'outputs/tokenizer')
        self.tokenizer_prefix = config.get('tokenizer_prefix', 'llm-tokenizer')
        self.normalization = config.get('normalization', 'NFC')
        
        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
        
    def _get_model_and_trainer(self):
        """
        Get the appropriate tokenizer model and trainer based on configuration.
        
        Returns:
            Tuple of (model, trainer) instances.
        """
        special_tokens = self.special_tokens
        
        if self.model_type == 'bpe':
            model = BPE(unk_token="<unk>" if "<unk>" in special_tokens else None)
            trainer = BpeTrainer(
                vocab_size=self.vocab_size,
                min_frequency=self.min_frequency,
                special_tokens=special_tokens,
                show_progress=True
            )
        elif self.model_type == 'unigram':
            model = Unigram()
            trainer = UnigramTrainer(
                vocab_size=self.vocab_size,
                special_tokens=special_tokens,
                show_progress=True
            )
        elif self.model_type == 'wordpiece':
            model = WordPiece(unk_token="<unk>" if "<unk>" in special_tokens else None)
            trainer = WordPieceTrainer(
                vocab_size=self.vocab_size,
                min_frequency=self.min_frequency,
                special_tokens=special_tokens,
                show_progress=True
            )
        elif self.model_type == 'char':
            model = WordLevel(unk_token="<unk>" if "<unk>" in special_tokens else None)
            trainer = WordLevelTrainer(
                vocab_size=self.vocab_size,
                min_frequency=self.min_frequency,
                special_tokens=special_tokens,
                show_progress=True
            )
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")
            
        return model, trainer
        
    def train(self, files: List[str]) -> Tokenizer:
        """
        Train a tokenizer on the given files.
        
        Args:
            files: List of file paths to train on.
            
        Returns:
            Trained tokenizer instance.
        """
        logger.info(f"Training {self.model_type} tokenizer with vocab size {self.vocab_size}")
        logger.info(f"Training on files: {files}")
        
        # Initialize the tokenizer with appropriate model
        model, trainer = self._get_model_and_trainer()
        tokenizer = Tokenizer(model)
        
        # Set up preprocessing
        if self.byte_level:
            tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
            logger.info("Using byte-level BPE")
        else:
            tokenizer.pre_tokenizer = Whitespace()
            logger.info("Using standard whitespace pre-tokenization")
            
        # Train the tokenizer
        logger.info("Starting tokenizer training...")
        tokenizer.train(files=files, trainer=trainer)
        logger.info(f"Finished training with {tokenizer.get_vocab_size()} tokens")
        
        # Set up post-processing
        if self.model_type == 'bpe' and self.byte_level:
            tokenizer.decoder = decoders.ByteLevel()
            
        # Add special tokens processing
        if "<s>" in self.special_tokens and "</s>" in self.special_tokens:
            logger.info("Adding BOS/EOS processing for special tokens")
            bos_id = tokenizer.token_to_id("<s>")
            eos_id = tokenizer.token_to_id("</s>")
            tokenizer.post_processor = processors.TemplateProcessing(
                single=f"<s>:0 $A:0 </s>:0",
                pair=f"<s>:0 $A:0 </s>:0 $B:1 </s>:1",
                special_tokens=[
                    ("<s>", bos_id),
                    ("</s>", eos_id),
                ],
            )
            
        return tokenizer
    
    def save_tokenizer(self, tokenizer: Tokenizer) -> None:
        """
        Save the trained tokenizer.
        
        Args:
            tokenizer: Trained tokenizer instance.
        """
        # Save the raw tokenizer files
        output_path = os.path.join(self.output_dir, self.tokenizer_prefix)
        logger.info(f"Saving tokenizer to {output_path}")
        tokenizer.save(f"{output_path}.json")
        
        # Also save in Hugging Face format for easier loading
        hf_tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=tokenizer,
            bos_token="<s>" if "<s>" in self.special_tokens else None,
            eos_token="</s>" if "</s>" in self.special_tokens else None,
            unk_token="<unk>" if "<unk>" in self.special_tokens else None,
            pad_token="<pad>" if "<pad>" in self.special_tokens else None,
            mask_token="<mask>" if "<mask>" in self.special_tokens else None,
            model_max_length=self.config.get('max_length', 2048),
            padding_side="right"
        )
        
        hf_tokenizer.save_pretrained(self.output_dir)
        logger.info(f"Saved HuggingFace tokenizer to {self.output_dir}")
        
        # Save vocabulary size info
        vocab_size = tokenizer.get_vocab_size()
        with open(os.path.join(self.output_dir, "vocab_size.txt"), "w") as f:
            f.write(str(vocab_size))
            
        logger.info(f"Tokenizer trained with final vocabulary size: {vocab_size}")
        
    def train_and_save(self) -> None:
        """
        Train and save a tokenizer using the configuration.
        """
        train_files = [self.config['train_data']]
        
        # Verify input files exist
        for file_path in train_files:
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"Training file not found: {file_path}")
        
        # Train tokenizer
        tokenizer = self.train(train_files)
        
        # Save the tokenizer
        self.save_tokenizer(tokenizer)
        
        # Save config for reproducibility
        import json
        with open(os.path.join(self.output_dir, "tokenizer_config.json"), "w") as f:
            json.dump(self.config, f, indent=2)
            
        logger.info("Tokenizer training complete")
        
    def load_tokenizer(self, fast: bool = True) -> PreTrainedTokenizerFast:
        """
        Load a trained tokenizer.
        
        Args:
            fast: Whether to load the fast tokenizer.
            
        Returns:
            Loaded tokenizer.
        """
        tokenizer_path = self.output_dir
        logger.info(f"Loading tokenizer from {tokenizer_path}")
        
        tokenizer = PreTrainedTokenizerFast.from_pretrained(tokenizer_path)
        
        vocab_size = tokenizer.vocab_size
        logger.info(f"Loaded tokenizer with vocabulary size: {vocab_size}")
        
        return tokenizer 