# entity/config_entity.py

from dataclasses import dataclass
from pathlib import Path
from typing import List

@dataclass(frozen=True)
class DataInjestionConfig:
    root_dir: Path
    source_URL: str
    local_data_file: Path
    unzip_dir: Path

@dataclass(frozen=True)
class DataValidationConfig:
    root_dir: Path
    data_path: Path
    status_file: Path        # where to write pass/fail result
    required_columns: List[str]  # columns that must exist
    min_rows: int            # minimum acceptable rows
    min_rating: float        # valid rating floor
    max_rating: float        # valid rating ceiling

@dataclass(frozen=True)
class DataTransformationConfig:
    root_dir: Path
    data_path: Path
    artifacts_dir: Path

@dataclass(frozen=True)
class ModelTrainerConfig:
    root_dir: Path
    data_path: Path
    n_neighbors: int
    metric: str
    algorithm: str