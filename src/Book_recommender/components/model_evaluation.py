import json
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity
import joblib  # add this
from Book_recommender.logging.log import logger
from Book_recommender.entity.config_entity import ModelEvaluationConfig

class ModelEvaluation:
    def __init__(self, config: ModelEvaluationConfig):
        self.config = config

    def load_artifacts(self):
        self.knn = joblib.load("artifacts/model_trainer/knn_model.pkl")       
        self.tfidf_matrix = joblib.load("artifacts/data_transformation/tfidf_matrix.pkl")  

        self.data = pd.read_csv(self.config.data_path)
        logger.info("Artifacts loaded for evaluation")

    def precision_at_k(self, book_idx, k):
        distances, indices = self.knn.kneighbors(
            self.tfidf_matrix[book_idx],
            n_neighbors=k + 1
        )
        rec_indices = indices.flatten()[1:]

        seed_genre = self.data.iloc[book_idx]["genre_clean"]
        rec_genres = self.data.iloc[rec_indices]["genre_clean"]

        # stronger check — covers NaN, None, float, empty string
        if not isinstance(seed_genre, str) or seed_genre.strip() == "":
            return None

        matches = rec_genres.str.contains(seed_genre, case=False, na=False, regex = False).sum()

        return matches / k

    def diversity_score(self, book_idx, k):
        """
        Measures how different recommendations are from each other
        1.0 = all different, 0.0 = all identical
        """
        distances, indices = self.knn.kneighbors(
            self.tfidf_matrix[book_idx],
            n_neighbors=k + 1
        )
        rec_indices = indices.flatten()[1:]
        rec_vectors = self.tfidf_matrix[rec_indices]

        sim_matrix = cosine_similarity(rec_vectors)
        # average similarity excluding diagonal
        np.fill_diagonal(sim_matrix, 0)
        avg_sim = sim_matrix.sum() / (k * (k - 1))

        return 1 - avg_sim  # higher = more diverse

    def coverage(self):
        """
        What % of books appear in at least one recommendation list
        """
        sample = self.data.sample(
            min(self.config.sample_size, len(self.data)),
            random_state=42
        )
        recommended = set()

        for idx in sample.index:
            _, indices = self.knn.kneighbors(
                self.tfidf_matrix[idx],
                n_neighbors=self.config.n_recommendations + 1
            )
            recommended.update(indices.flatten()[1:].tolist())

        return len(recommended) / len(self.data)

    def run_evaluation(self):
        self.load_artifacts()

        sample = self.data.sample(
            min(self.config.sample_size, len(self.data)),
            random_state=42
        )

        precisions = []
        diversities = []

        for idx in sample.index:
            p = self.precision_at_k(idx, self.config.n_recommendations)
            d = self.diversity_score(idx, self.config.n_recommendations)

            if p is not None:
                precisions.append(p)
            diversities.append(d)

        metrics = {
            "precision_at_k":   round(np.mean(precisions), 4),
            "diversity_score":  round(np.mean(diversities), 4),
            "catalog_coverage": round(self.coverage(), 4),
            "k":                self.config.n_recommendations,
            "sample_size":      len(sample)
        }

        # save metrics
        with open(self.config.metrics_file, "w") as f:
            json.dump(metrics, f, indent=4)

        logger.info(f"Evaluation complete: {metrics}")
        return metrics