import os
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from Book_recommender.logging.log import logger
import pandas as pd

#Component
class DataTransformation:
    def __init__(self, config):
        self.config = config
        self.vectorizer = None

    def initialize_vectorizer(self):
        self.vectorizer = TfidfVectorizer(
            stop_words="english",
            max_features=10000,
            ngram_range=(1, 2),
            min_df=2
        )

    def fit_transform(self, data):
        return self.vectorizer.fit_transform(
            data["combined_features"]
        )

    def save(self, path):

        df = pd.read_csv(self.config.data_path)
        # save vectorizer
        joblib.dump(self.vectorizer, path)
        logger.info(f"Vectorizer saved to {path}")

        # save tfidf matrix separately
        tfidf_matrix = self.vectorizer.transform(df["combined_features"])
        joblib.dump(tfidf_matrix, "artifacts/data_transformation/tfidf_matrix.pkl")
        logger.info("TF-IDF matrix saved")