
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from datasets import load_dataset, load_from_disk
import torch
import evaluate
from evaluate import load
import pandas as pd
from tqdm import tqdm
from textSummarizer.entity import ModelEvaluationConfig



import os
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from datasets import load_from_disk
import pandas as pd
import torch
import evaluate  # Import the evaluate module instead of load_metric
import logging
from tqdm import tqdm


class ModelEvaluation:
    def __init__(self, config: ModelEvaluationConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        logging.basicConfig(level=logging.INFO)

    def generate_batch_sized_chunks(self, list_of_elements, batch_size):
        """Split the dataset into smaller batches that we can process simultaneously.
        Yield successive batch-sized chunks from list_of_elements."""
        for i in range(0, len(list_of_elements), batch_size):
            yield list_of_elements[i: i + batch_size]

    def calculate_metric_on_train_ds(self, dataset, metric, model, tokenizer, 
                                     batch_size=16, device="cuda" if torch.cuda.is_available() else "cpu", 
                                     column_text="article", 
                                     column_summary="highlights"):
        article_batches = list(self.generate_batch_sized_chunks(dataset[column_text], batch_size))
        target_batches = list(self.generate_batch_sized_chunks(dataset[column_summary], batch_size))

        for article_batch, target_batch in tqdm(
            zip(article_batches, target_batches), total=len(article_batches)):
            
            # Tokenize the article batch
            inputs = tokenizer(article_batch, max_length=1024, truncation=True, 
                               padding="max_length", return_tensors="pt")
            
            # Generate summaries using the model
            summaries = model.generate(input_ids=inputs["input_ids"].to(device),
                                       attention_mask=inputs["attention_mask"].to(device), 
                                       length_penalty=0.8, num_beams=8, max_length=128)
            '''Parameter for length penalty ensures that the model does not generate sequences that are too long.'''
            
            # Decode the generated texts, clean up tokenization spaces
            decoded_summaries = [tokenizer.decode(s, skip_special_tokens=True, 
                                                  clean_up_tokenization_spaces=True) 
                                 for s in summaries]
            
            decoded_summaries = [d.replace("", " ") for d in decoded_summaries]
            
            # Add the predictions and references to the metric
            metric.add_batch(predictions=decoded_summaries, references=target_batch)
            
        # Compute and return the ROUGE scores
        score = metric.compute()
        return score

    def evaluate(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Use absolute path to the tokenizer directory
        tokenizer_path = os.path.abspath(self.config.tokenizer_path)
        
        # Ensure the tokenizer path exists
        if not os.path.exists(tokenizer_path):
            raise ValueError(f"Tokenizer path does not exist: {tokenizer_path}")
        
        # Load the tokenizer and model
        self.logger.info("Loading tokenizer and model...")
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        model_t5 = AutoModelForSeq2SeqLM.from_pretrained(self.config.model_path).to(device)

        # Load the dataset
        self.logger.info("Loading dataset...")
        dataset_samsum_pt = load_from_disk(self.config.data_path)

        # Subset the dataset (e.g., select the first 100 examples from the validation set)
        subset_size = 100  # Adjust this to the desired size
        tokenized_eval_dataset = dataset_samsum_pt["validation"].select(range(subset_size))  # Take the first `subset_size` examples

        # Calculate ROUGE scores
        rouge_metric = evaluate.load("rouge")

        self.logger.info("Calculating ROUGE scores...")
        score = self.calculate_metric_on_train_ds(
            dataset=tokenized_eval_dataset,
            metric=rouge_metric,
            model=model_t5,
            tokenizer=tokenizer,
            batch_size=2,
            column_text="dialogue",
            column_summary="summary"
        )

        # Format the results for saving
        rouge_dict = {rn: score[rn] for rn in ["rouge1", "rouge2", "rougeL", "rougeLsum"]}  # Directly use the scores

        # Save results to a CSV file in the model_evaluation folder
        output_path = os.path.join(self.config.root_dir, "metrics.csv")  # Save in the artifacts/model_evaluation directory
        df = pd.DataFrame([rouge_dict])  # Wrap in a list to create a single-row DataFrame
        df.to_csv(output_path, index=False)

        self.logger.info(f"Evaluation metrics saved to {output_path}")
