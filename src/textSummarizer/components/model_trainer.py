import os
from transformers import TrainingArguments, Trainer
from transformers import DataCollatorForSeq2Seq
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from datasets import load_from_disk
from textSummarizer.entity import ModelTrainerConfig
from textSummarizer.logging import logger
from pathlib import Path
import torch

class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig):
        self.config = config

    def freeze_layers(self, model):
        """Freeze encoder and decoder layers to speed up training."""
        for name, param in model.named_parameters():
            if 'encoder' in name or 'decoder' in name:  # Freeze encoder and decoder layers
                param.requires_grad = False

    def train(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {device}")

        # Load tokenizer and model
        tokenizer = AutoTokenizer.from_pretrained(self.config.model_ckpt)
        model_pegasus = AutoModelForSeq2SeqLM.from_pretrained(self.config.model_ckpt).to(device)

        logger.info("Model and tokenizer loaded successfully.")

        # Freeze encoder and decoder layers
        self.freeze_layers(model_pegasus)
        logger.info("Frozen encoder and decoder layers to speed up training.")

        # Data Collator for Seq2Seq
        seq2seq_data_collator = DataCollatorForSeq2Seq(tokenizer, model=model_pegasus)

        # Load dataset
        dataset_samsum_pt = load_from_disk(self.config.data_path)
        logger.info(f"Dataset loaded from {self.config.data_path}.")

        # Use only a subset of the data (For debugging purposes, consider using more data in real training)
        train_dataset = dataset_samsum_pt["train"].select(range(6))  # Use first 2 samples for training
        eval_dataset = dataset_samsum_pt["validation"].select(range(2))  # Use first 2 samples for validation

        logger.info("Datasets prepared for training and evaluation.")

        # Define training arguments
        trainer_args = TrainingArguments(
            output_dir=str(self.config.root_dir),
            num_train_epochs=0.1,  # Small number of epochs for debugging
            per_device_train_batch_size=1,
            per_device_eval_batch_size=1,
            save_steps=1000,
            save_total_limit=1,
            evaluation_strategy='steps',
            logging_steps=500,
            gradient_accumulation_steps=8,
            fp16=True,  # Mixed precision
            remove_unused_columns=True,
            load_best_model_at_end=True,
            dataloader_num_workers=4,
            run_name="optimized_training_run"
        )

        logger.info("Training arguments defined.")

        # Initialize Trainer
        trainer = Trainer(
            model=model_pegasus,
            args=trainer_args,
            tokenizer=tokenizer,
            data_collator=seq2seq_data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset
        )

        logger.info("Trainer initialized.")

        # Train the model
        trainer.train()
        logger.info("Model training completed.")

        # Save model and tokenizer
        model_save_path = os.path.join(self.config.root_dir, "pegasus-samsum-model")
        tokenizer_save_path = os.path.join(self.config.root_dir, "tokenizer")
        model_pegasus.save_pretrained(model_save_path)
        tokenizer.save_pretrained(tokenizer_save_path)

        logger.info(f"Model saved to {model_save_path}.")
        logger.info(f"Tokenizer saved to {tokenizer_save_path}.")


'''
import os
from transformers import TrainingArguments, Trainer
from transformers import DataCollatorForSeq2Seq
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from datasets import load_from_disk
from textSummarizer.entity import ModelTrainerConfig
from textSummarizer.logging import logger
from pathlib import Path
import torch

class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig):
        self.config = config

    def freeze_layers(self, model):
        """Freeze encoder and decoder layers to speed up training."""
        for name, param in model.named_parameters():
            if 'encoder' in name or 'decoder' in name:  # Freeze encoder and decoder layers
                param.requires_grad = False

    def train(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Load tokenizer and model
        tokenizer = AutoTokenizer.from_pretrained(self.config.model_ckpt)
        model_pegasus = AutoModelForSeq2SeqLM.from_pretrained(self.config.model_ckpt).to(device)

        # Freeze encoder and decoder layers
        self.freeze_layers(model_pegasus)

        # Data Collator for Seq2Seq
        seq2seq_data_collator = DataCollatorForSeq2Seq(tokenizer, model=model_pegasus)

        # Load dataset
        dataset_samsum_pt = load_from_disk(self.config.data_path)

        # Use only a subset of the data (For debugging purposes, consider using more data in real training)
        train_dataset = dataset_samsum_pt["train"].select(range(6))  # Use first 6 samples for training
        eval_dataset = dataset_samsum_pt["validation"].select(range(2))  # Use first 2 samples for validation

        # Define training arguments
        trainer_args = TrainingArguments(
            output_dir=str(self.config.root_dir),  # Convert to string path
            num_train_epochs=0.1,  # Small number of epochs for debugging
            per_device_train_batch_size=1,
            per_device_eval_batch_size=1,
            save_steps=1000,
            save_total_limit=1,
            evaluation_strategy='steps',
            logging_steps=500,
            gradient_accumulation_steps=8,
            fp16=True,  # Mixed precision
            remove_unused_columns=True,
            load_best_model_at_end=True,  # Save best model during training
            dataloader_num_workers=4,  # Parallelize data loading if possible
            run_name="optimized_training_run"  # Optional: Track the run in a specific way
        )

        # Initialize Trainer
        trainer = Trainer(
            model=model_pegasus,
            args=trainer_args,
            tokenizer=tokenizer.processing_class,
            data_collator=seq2seq_data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset
        )

        # Train the model
        trainer.train()

        # Save model and tokenizer
        model_pegasus.save_pretrained(os.path.join(self.config.root_dir, "pegasus-samsum-model"))
        tokenizer.save_pretrained(os.path.join(self.config.root_dir, "tokenizer"))
'''