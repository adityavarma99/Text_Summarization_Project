from textSummarizer.config.configuration import ConfigurationManager
from textSummarizer.components.model_trainer import ModelTrainer
from textSummarizer.logging import logger


class ModelTrainerTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        try:
            # Initialize configuration manager
            config = ConfigurationManager()

            # Get model trainer configuration
            model_trainer_config = config.get_model_trainer_config()

            # Initialize and execute the ModelTrainer
            model_trainer = ModelTrainer(config=model_trainer_config)
            logger.info("Starting model training...")
            model_trainer.train()
            logger.info("Model training completed successfully.")

        except Exception as e:
            logger.error(f"Error occurred in model trainer pipeline: {e}")
            raise
