from Book_recommender.config.configuration import configurationManager
from Book_recommender.components.data_validation import Datavalidation
from Book_recommender.components.data_processing import DataProcessing
from Book_recommender.logging.log import logger
import pandas as pd

class DataProcessingTrainingPipeline:
    def __init__ (self):
        pass

    def main(self):
        config = configurationManager()
        data_preprocessing_config = config.get_data_preprocessing_config()
        
        data = pd.read_csv("artifacts/data_injection/data.csv")  # load your data
        
        data_processing = DataProcessing(config=data_preprocessing_config)
        data_processing.process(data)