from sklearn.neighbors import NearestNeighbors
import joblib

from Book_recommender.components.neighbor_models import SvdNeighbors
from Book_recommender.logging.log import logger

MODEL_REGISTRY = {
    "knn": NearestNeighbors,
    "svd_knn": SvdNeighbors,
}


class ModelTrainer:
    def __init__(self, config):
        self.config = config
        self.model = None

    def initialize_model(self):
        model_cls = MODEL_REGISTRY[self.config.model_name]
        self.model = model_cls(**self.config.model_params)
        logger.info(f"Initialised {self.config.model_name} with {self.config.model_params}")

    def fit_model(self, tfidf_matrix):
        self.model.fit(tfidf_matrix)

    def save(self):
        joblib.dump(self.model, self.config.model_path)