import joblib
from sklearn.feature_extraction.text import TfidfVectorizer

from Book_recommender.logging.log import logger


class DataTransformation:
    def __init__(self, config):
        self.config = config
        self.vectorizer = None

    def initialize_vectorizer(self):
        params = dict(self.config.tfidf_params)
        # YAML gives a list; scikit-learn wants a tuple
        params["ngram_range"] = tuple(params["ngram_range"])
        self.vectorizer = TfidfVectorizer(**params)
        logger.info(f"TF-IDF params: {params}")

    def fit_transform(self, data):
        text = data["combined_features"].fillna("")
        matrix = self.vectorizer.fit_transform(text)
        logger.info(f"TF-IDF matrix shape: {matrix.shape}")
        return matrix

    def save(self, tfidf_matrix):
        joblib.dump(self.vectorizer, self.config.vectorizer_path)
        joblib.dump(tfidf_matrix, self.config.tfidf_matrix_path)
        logger.info("Vectorizer and TF-IDF matrix saved")