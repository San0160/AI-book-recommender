from pathlib import Path
import yaml
import sys
import os

from box import ConfigBox
from ensure import ensure_annotations

from Book_recommender.logging.log import logging
from Book_recommender.exception.exception_handler import CustomException


@ensure_annotations
def read_yaml(path_to_yaml: Path) -> ConfigBox:
    """
    Reads YAML file and returns ConfigBox object.

    Args:
        path_to_yaml (Path): Path to YAML file

    Returns:
        ConfigBox: Parsed YAML content
    """

    try:
        with open(path_to_yaml, "r") as yaml_file:

            content = yaml.safe_load(yaml_file)

            if content is None:
                raise ValueError("YAML file is empty")

            logging.info(f"YAML file loaded successfully from: {path_to_yaml}")

            return ConfigBox(content)

    except Exception as e:
        logging.error(f"Failed to read YAML file: {path_to_yaml}")

        raise CustomException(e, sys) from e
    

@ensure_annotations
def create_directories(path_to_directories: list, verbose = True):
    """create list of directories
    
    args:
        path to directories (list) : list of path to directories
        ignore_log (bool, optional): ignore if multiple directories is to be created. default to false
    """
    for path in path_to_directories:
        os.makedirs(path, exist_ok=True)
        if verbose:
            logging.info(f"created directory at {path}")


@ensure_annotations
def get_size(path: Path) -> str:
    """get size in kb

    args:
        path (Path): path of the file
        
    Return:
        str: size in KB
    """
    size_in_kb = round(os.path.getsize(path)/1024)
    return f"~ {size_in_kb} KB"