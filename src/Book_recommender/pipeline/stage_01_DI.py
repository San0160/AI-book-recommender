from Book_recommender.config.configuration import configurationManager
from Book_recommender.components.data_injection import DataInjection
from Book_recommender.logging.log import logger

class DataInjectionTrainingPipeline:
    def __init__ (self):
        pass

    def main(self):
        config = configurationManager()
        data_injection_config = config.get_data_injection_config()
        data_injection = DataInjection(config = data_injection_config)
        data_injection.download_files()
        data_injection.extract_zipfile()