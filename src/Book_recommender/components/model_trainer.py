from sklearn.neighbors import NearestNeighbors
import joblib

class ModelTrainer:
    def __init__(self, config):
        self.config = config
        self.model = None

    def initialize_model(self):
        if self.config.model_name != "knn":
            raise NotImplementedError(
                f"'{self.config.model_name}' is added in Step 3"
            )
        self.model = NearestNeighbors(**self.config.model_params)

    def fit_model(self, tfidf_matrix):
        self.model.fit(tfidf_matrix)

    def save(self):
        joblib.dump(
        self.model,
        self.config.model_path
    )