import joblib
from sklearn.feature_extraction.text import TfidfVectorizer

from Book_recommender.logging.log import logger

SUPPORTED_TRANSFORMS = ("tfidf", "embeddings")


class DataTransformation:
    def __init__(self, config):
        self.config = config
        self.vectorizer = None  # only set for tfidf

    def fit_transform(self, data):
        text = data["combined_features"].fillna("").tolist()
        if self.config.method == "tfidf":
            matrix = self._tfidf(text)
        elif self.config.method == "embeddings":
            matrix = self._embeddings(text)
        else:
            raise ValueError(f"Unknown method: {self.config.method}")
        logger.info(f"{self.config.method} feature matrix shape: {matrix.shape}")
        return matrix

    def _tfidf(self, text):
        params = dict(self.config.method_params)
        params["ngram_range"] = tuple(params["ngram_range"])  # YAML list -> tuple
        self.vectorizer = TfidfVectorizer(**params)
        return self.vectorizer.fit_transform(text)

    def _embeddings(self, text):
        # Imported here so TF-IDF runs and the API never need PyTorch loaded
        from sentence_transformers import SentenceTransformer

        p = self.config.method_params
        model = SentenceTransformer(p["model_name"])
        return model.encode(
            text,
            batch_size=p["batch_size"],
            normalize_embeddings=p["normalize"],
            show_progress_bar=True,
            convert_to_numpy=True,
        ).astype("float32")

    def save(self, matrix):
        joblib.dump(matrix, self.config.feature_matrix_path)
        if self.vectorizer is not None:
            joblib.dump(self.vectorizer, self.config.vectorizer_path)
        logger.info("Feature matrix saved")