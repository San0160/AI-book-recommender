from Book_recommender.config.configuration import configurationManager
from Book_recommender.components.data_validation import Datavalidation
from Book_recommender.components.data_processing import DataProcessing
from Book_recommender.components.data_transformation import DataTransformation
from Book_recommender.logging.log import logger
import pandas as pd

class DataTransformationTrainingPipeline:
    def __init__ (self):
        pass

    def main(self):
        config = configurationManager()
        data_transformation_config = config.get_data_transformation_config()
        
        data = pd.read_csv("artifacts/data_preprocessing/processed_data.csv")  # load your data
        
        data_processing = DataTransformation(config=data_transformation_config)
        data_processing.initialize_vectorizer()
        data_processing.fit_transform(data)
        data_processing.save(data_transformation_config.vectorizer_path)