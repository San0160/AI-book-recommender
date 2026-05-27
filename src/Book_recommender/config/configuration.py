from Book_recommender.constant import *
from Book_recommender.utils.common import read_yaml, create_directories
from Book_recommender.entity.config_entity import (DataInjestionConfig)

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
            local_data_file = config.local_data_file,
            unzip_dir = config.unzip_dir
        )

        return data_injection_config