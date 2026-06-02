from Book_recommender.constant import *
from Book_recommender.utils.common import read_yaml, create_directories
from Book_recommender.entity.config_entity import (DataInjestionConfig,
                                                   DatavalidationConfig,
                                                   DataProcessingConfig,
                                                   DataTransformationConfig,
                                                   ModelTrainerConfig,
                                                   ModelEvaluationConfig)

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
    
    def get_data_transformation_config(self) -> DataTransformationConfig:
        config = self.config.data_transformation

        create_directories([config.root_dir])

        data_transformation_config = DataTransformationConfig(
            root_dir = config.root_dir,
            data_path=config.data_path,
            vectorizer_path= config.vectorizer_path
        )

        return data_transformation_config
    
    def get_model_trainer_config(self) -> ModelTrainerConfig:
        config = self.config.model_trainer

        create_directories([config.root_dir])

        model_trainer_config = ModelTrainerConfig(
            root_dir = config.root_dir,
            data_path = config.data_path,
            vectorizer_path = config.vectorizer_path,
            model_path = config.model_path,
            n_neighbors = config.n_neighbors,
            metric = config.metric,
            algorithm = config.algorithm
        )
        return model_trainer_config
    
    def get_model_evaluation_config(self) -> ModelEvaluationConfig:
        config = self.config.model_evaluation

        create_directories([config.root_dir])

        model_evaluation_config = ModelEvaluationConfig(  
            root_dir=config.root_dir,
            data_path=config.data_path,
            model_path=config.model_path,
            metrics_file = config.metrics_file,
            sample_size = config.sample_size,     
            n_recommendations =  config.n_recommendations
        )

        return model_evaluation_config
        