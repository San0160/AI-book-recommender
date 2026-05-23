from pathlib import Path
import yaml
import sys

from box import ConfigBox
from ensure import ensure_annotations

from Book_recommender.logging.log import logging
from Book_recommender.exception.exception import CustomException


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