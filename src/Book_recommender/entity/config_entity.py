# entity/config_entity.py

from dataclasses import dataclass
from pathlib import Path
from typing import List

@dataclass(frozen=True)
class DataInjestionConfig:
    root_dir: Path
    source_URL: str
    local_data_file: Path

@dataclass(frozen = True)
class DatavalidationConfig:
    root_dir: Path
    STATUS_FILE: str
    ALL_REQUIRED_FILES: list

@dataclass(frozen = True)
class DataProcessingConfig:
    root_dir: Path
    local_data_file: Path
    feature_fields: list

@dataclass(frozen=True)
class DataTransformationConfig:
    root_dir: Path
    data_path: Path
    vectorizer_path: Path
    feature_matrix_path: Path
    method: str
    method_params: dict

@dataclass(frozen=True)
class ModelTrainerConfig:
    root_dir: Path
    data_path: Path
    feature_matrix_path: Path
    model_path: Path
    model_name: str
    model_params: dict

@dataclass(frozen=True)
class ModelEvaluationConfig:
    root_dir: Path
    data_path: Path
    model_path: Path
    metrics_file: Path
    sample_size: int      
    n_recommendations: int 
    feature_matrix_path: Path
    mlflow_tracking_uri: str
    mlflow_experiment_name: str