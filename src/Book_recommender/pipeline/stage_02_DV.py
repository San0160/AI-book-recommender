from Book_recommender.config.configuration import configurationManager
from Book_recommender.components.data_validation import Datavalidation
from Book_recommender.logging.log import logger

class DataValidationTrainingPipeline:
    def __init__ (self):
        pass

    def main(self):
        config = configurationManager()
        data_validation_config = config.get_data_validation_config()
        data_validation = Datavalidation(config = data_validation_config)
        data_validation.validate_all_files_exists()