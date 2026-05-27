import os
import urllib.request as request
import zipfile
from Book_recommender.logging.log import logger
from Book_recommender.utils.common import get_size
from Book_recommender.entity.config_entity import DataInjestionConfig
from pathlib import Path

# 5 Conponents

class DataInjection:
    def __init__(self, config: DataInjestionConfig):
        self.config = config


    def download_files(self):
        if not os.path.exists(self.config.local_data_file):
            filename, headers = request.urlretrieve(
                url = self.config.source_URL,
                filename = self.config.local_data_file
            )
            logger.info(f"{filename} Downloaded with following info \n{headers}")
        else:
            logger.info(f"file already exist of size: {get_size(Path(self.config.local_data_file))}")

    def extract_zipfile(self):
        """
        zip_file_path: str
        Extracts the zip file in dir
        None returns
        """
        unzip_path = self.config.unzip_dir
        os.makedirs(unzip_path, exist_ok = True)
        with zipfile.ZipFile(self.config.local_data_file, "r") as zip_ref:
            zip_ref.extractall(unzip_path)