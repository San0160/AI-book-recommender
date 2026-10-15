from Book_recommender.config.configuration import configurationManager
from Book_recommender.components.data_processing import DataProcessing
import pandas as pd

class DataProcessingTrainingPipeline:
    def __init__ (self):
        pass

    def main(self):
        config = configurationManager()
        data_processing_config = config.get_data_preprocessing_config()
        
        data = pd.read_csv(data_processing_config.data_path)
        
        data_processing = DataProcessing(config=data_processing_config)
        data_processing.process(data)