from Book_recommender.config.configuration import configurationManager
from Book_recommender.components.model_evaluation import ModelEvaluation
from Book_recommender.logging.log import logger
import pandas as pd
import os

class ModelEvaulationTrainingPipeline:
    def __init__ (self):
        pass

    def main(self):
        config = configurationManager()
        model_evaluation_config = config.get_model_evaluation_config()
        
        if not os.path.exists(model_evaluation_config.metrics_file):
            logger.info("No metrics found — running evaluation...")
            model_evaluation = ModelEvaluation(config=model_evaluation_config)
            metrics = model_evaluation.run_evaluation()
            print(metrics)
        else:
            logger.info("Metrics already exist — skipping evaluation!")