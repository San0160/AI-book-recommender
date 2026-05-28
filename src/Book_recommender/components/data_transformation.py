import os
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer

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
        joblib.dump(self.vectorizer, path)