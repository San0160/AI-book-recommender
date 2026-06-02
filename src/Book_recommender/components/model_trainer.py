from sklearn.neighbors import NearestNeighbors
import joblib

class ModelTrainer:
    def __init__(self, config):
        self.config = config
        self.model = None

    def initialize_model(self):
        self.model = NearestNeighbors(
        n_neighbors=self.config.n_neighbors,
        metric=self.config.metric,
        algorithm=self.config.algorithm
    )

    def fit_model(self, tfidf_matrix):
        self.model.fit(tfidf_matrix)

    def save(self):
        joblib.dump(
        self.model,
        self.config.model_path
    )