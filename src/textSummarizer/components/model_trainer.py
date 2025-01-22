import os
from transformers import TrainingArguments, Trainer
from transformers import DataCollatorForSeq2Seq
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from datasets import load_from_disk
from textSummarizer.entity import ModelTrainerConfig
from textSummarizer.logging import logger
from pathlib import Path
import torch

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Trainer, TrainingArguments, DataCollatorForSeq2Seq
import torch
import os
from datasets import load_from_disk

class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig):
        self.config = config

    def train(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Load T5 model and tokenizer
        tokenizer = AutoTokenizer.from_pretrained("t5-small")  # T5 tokenizer
        model_t5 = AutoModelForSeq2SeqLM.from_pretrained("t5-small").to(device)
        seq2seq_data_collator = DataCollatorForSeq2Seq(tokenizer, model=model_t5)

        # Load dataset (assuming it's stored in the correct format)
        dataset_samsum_pt = load_from_disk(self.config.data_path)
        
        # Select a larger sample for training
        train_dataset = dataset_samsum_pt["train"].select(range(100))
        eval_dataset = dataset_samsum_pt["validation"].select(range(100))  # Use the validation split

        # Tokenize the dataset
        def tokenize_function(examples):
            # Ensure input and labels are tokenized with padding and truncation
            model_inputs = tokenizer(
                examples["dialogue"],
                max_length=1024,
                padding="max_length",
                truncation=True,
            )
            
            # Tokenize summaries as labels
            with tokenizer.as_target_tokenizer():
                labels = tokenizer(
                    examples["summary"], 
                    max_length=128, 
                    padding="max_length", 
                    truncation=True
                )
                
            model_inputs["labels"] = labels["input_ids"]
            return model_inputs

        # Apply tokenization
        tokenized_train_dataset = train_dataset.map(tokenize_function, batched=True)
        tokenized_eval_dataset = eval_dataset.map(tokenize_function, batched=True)
        
        # Set parameters for training
        trainer_args = TrainingArguments(
            output_dir=self.config.root_dir,  # where the model checkpoints will be saved
            num_train_epochs=3,  # Increased epochs
            warmup_steps=500,  # Increase warmup steps for smoother learning
            per_device_train_batch_size=2,  # Increased batch size to 2
            per_device_eval_batch_size=2,  # Increased eval batch size to 2
            weight_decay=0.01,  # Small weight decay for regularization
            logging_steps=10,  # Log every 10 steps to reduce log frequency
            evaluation_strategy='steps',  # Perform evaluation every `eval_steps`
            eval_steps=500,  # Perform evaluation every 500 steps
            save_steps=500,  # Save checkpoints every 500 steps
            gradient_accumulation_steps=2,  # Accumulate gradients over 2 steps
            fp16=False,  # Disable mixed precision (don't need for low-end systems)
            load_best_model_at_end=False,  # Don't worry about the best model
            logging_dir='./logs',  # Optional, directory to store logs
            report_to=[]  # Disable reporting to external systems like TensorBoard
        )

        # Initialize Trainer
        trainer = Trainer(
            model=model_t5,
            args=trainer_args,
            tokenizer=tokenizer,
            data_collator=seq2seq_data_collator,
            train_dataset=tokenized_train_dataset,  # Use tokenized train dataset
            eval_dataset=tokenized_eval_dataset  # Use tokenized eval dataset for evaluation
        )
        
        # Start training
        trainer.train()

        # Save the model and tokenizer (optional)
        model_save_path=model_t5.save_pretrained(os.path.join(self.config.root_dir, "t5-samsum-model"))
        tokenizer_save_path=tokenizer.save_pretrained(os.path.join(self.config.root_dir, "tokenizer"))


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