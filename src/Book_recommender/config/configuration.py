from Book_recommender.constant import *
from Book_recommender.utils.common import read_yaml, create_directories
from Book_recommender.entity.config_entity import (DataInjestionConfig, DatavalidationConfig, DataProcessingConfig)

# 4 Update configuration manager

class configurationManager:
    def __init__(
        self,
        config_filepath = CONFIG_FILE_PATH,     # Access to constants
        params_filepath = PARAMS_FILE_PATH):

        self.config = read_yaml(config_filepath) # read all config and params yaml files
        self.params = read_yaml(params_filepath)

        create_directories([self.config.artifacts_root])

    def get_data_injection_config(self) -> DataInjestionConfig:
        config = self.config.data_injection

        create_directories([config.root_dir])

        data_injection_config = DataInjestionConfig(
            root_dir = config.root_dir,
            source_URL = config.source_URL,
            local_data_file = config.local_data_file
        )

        return data_injection_config
    
    def get_data_validation_config(self) -> DatavalidationConfig:
        config = self.config.data_validation

        create_directories([config.root_dir])

        data_validation_config = DatavalidationConfig(
            root_dir = config.root_dir,
            STATUS_FILE = config.STATUS_FILE,
            ALL_REQUIRED_FILES = config.ALL_REQUIRED_FILES
        )

        return data_validation_config
    

    def get_data_preprocessing_config(self) -> DataProcessingConfig:
        config = self.config.data_processing

        create_directories([config.root_dir])

        data_processing_config = DataProcessingConfig(
            root_dir = config.root_dir,
            local_data_file = config.local_data_file
        )

        return data_processing_config