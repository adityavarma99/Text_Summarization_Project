import os
from textSummarizer.logging import logger
from transformers import AutoTokenizer
from datasets import load_dataset, load_from_disk
from textSummarizer.entity import DataTransformationConfig

class DataTransformation:
    def __init__(self, config: DataTransformationConfig):
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name)

    def convert_examples_to_features(self, example_batch):
        input_encodings = self.tokenizer(
            example_batch['dialogue'],
            max_length=1024,
            truncation=True
        )

        with self.tokenizer.as_target_tokenizer():
            target_encodings = self.tokenizer(
                example_batch['summary'],
                max_length=128,
                truncation=True
            )
        
        return {
            'input_ids': input_encodings['input_ids'],
            'attention_mask': input_encodings['attention_mask'],
            'labels': target_encodings['input_ids']
        }

    def convert(self):
        try:
            # Ensure dataset path is correct, handle backward slashes
            if not os.path.exists(self.config.data_path):
                raise FileNotFoundError(f"The specified dataset path does not exist: {self.config.data_path}")
            
            # Load the dataset from disk
            dataset_samsum = load_from_disk(self.config.data_path)

            # Transform the dataset
            dataset_samsum_pt = dataset_samsum.map(self.convert_examples_to_features, batched=True)

            # Define the save path and handle backward slashes using os.path.join
            save_path = os.path.join(self.config.root_dir, "samsum_dataset")
            
            # Check if the path length exceeds typical limits
            if len(save_path) > 255:  # Adjust limit if necessary for your OS
                raise ValueError(f"File path is too long: {save_path}")
            
            # Ensure the directory exists
            os.makedirs(save_path, exist_ok=True)
            logger.info(f"Save directory ensured: {save_path}")

            # Save the transformed dataset to disk
            dataset_samsum_pt.save_to_disk(save_path)
            logger.info(f"Dataset saved successfully to {save_path}")

        except Exception as e:
            logger.error(f"Error during data transformation: {e}")
            raise



'''
import os
from textSummarizer.logging import logger
from transformers import AutoTokenizer
from datasets import load_dataset, load_from_disk
from textSummarizer.entity import DataTransformationConfig



class DataTransformation:
    def __init__(self, config: DataTransformationConfig):
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name)


    
    def convert_examples_to_features(self,example_batch):
        input_encodings = self.tokenizer(example_batch['dialogue'] , max_length = 1024, truncation = True )
        
        with self.tokenizer.as_target_tokenizer():
            target_encodings = self.tokenizer(example_batch['summary'], max_length = 128, truncation = True )
            
        return {
            'input_ids' : input_encodings['input_ids'],
            'attention_mask': input_encodings['attention_mask'],
            'labels': target_encodings['input_ids']
        }
    

    def convert(self):
        dataset_samsum = load_from_disk(self.config.data_path)
        dataset_samsum_pt = dataset_samsum.map(self.convert_examples_to_features, batched = True)
        dataset_samsum_pt.save_to_disk(os.path.join(self.config.root_dir,"samsum_dataset"))
'''
'''
import os
from textSummarizer.logging import logger
from transformers import AutoTokenizer
from datasets import load_dataset, load_from_disk
from textSummarizer.entity import DataTransformationConfig

class DataTransformation:
    def __init__(self, config: DataTransformationConfig):
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name)

    def convert_examples_to_features(self, example_batch):
        input_encodings = self.tokenizer(
            example_batch['dialogue'], 
            max_length=1024, 
            truncation=True
        )
        target_encodings = self.tokenizer(
            example_batch['summary'], 
            max_length=128, 
            truncation=True, 
            text_target=example_batch['summary']
        )
        return {
            'input_ids': input_encodings['input_ids'],
            'attention_mask': input_encodings['attention_mask'],
            'labels': target_encodings['input_ids']
        }

    def convert(self):
        try:
            # Load the dataset from disk
            if not os.path.exists(self.config.data_path):
                raise FileNotFoundError(f"The specified dataset path does not exist: {self.config.data_path}")
            
            dataset_samsum = load_from_disk(self.config.data_path)

            # Transform the dataset
            dataset_samsum_pt = dataset_samsum.map(self.convert_examples_to_features, batched=True)

            # Use the hard-coded save path
            save_path = r"E:\Data science\Text_summarization\artifacts\data_transformation\samsum_dataset"
            
            # Check if the path length exceeds typical limits
            if len(save_path) > 255:  # Adjust limit if necessary for your OS
                raise ValueError(f"File path is too long: {save_path}")
            
            # Create the directory if it does not exist
            os.makedirs(save_path, exist_ok=True)
            logger.info(f"Save directory ensured: {save_path}")

            # Save the transformed dataset to disk
            dataset_samsum_pt.save_to_disk(save_path)
            logger.info(f"Dataset saved successfully to {save_path}")

        except Exception as e:
            logger.error(f"Error during data transformation: {e}")
            raise
'''
