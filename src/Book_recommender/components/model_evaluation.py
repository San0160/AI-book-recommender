import json
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

from Book_recommender.logging.log import logger
from Book_recommender.entity.config_entity import ModelEvaluationConfig


class ModelEvaluation:
    def __init__(self, config: ModelEvaluationConfig):
        self.config = config

    def load_artifacts(self):
        self.model = joblib.load(self.config.model_path)
        self.features = joblib.load(self.config.feature_matrix_path)
        self.data = pd.read_csv(self.config.data_path).reset_index(drop=True)

        self.genres = (
            self.data["genre_clean"].fillna("").astype(str).str.strip().str.lower().to_numpy()
        )

        # Popularity ranking (same Bayesian score the API uses)
        v = pd.to_numeric(self.data["ratings_count"], errors="coerce").fillna(0)
        r = pd.to_numeric(self.data["average_rating"], errors="coerce").fillna(0)
        C, m = r.mean(), v.quantile(0.75)
        score = (v / (v + m)) * r + (m / (v + m)) * C
        self.popular_order = np.argsort(-score.to_numpy())

        logger.info(f"Evaluating {type(self.model).__name__}")

    # ---- three recommenders, same signature: (query_idx, k) -> array of indices
    def _model_recs(self, idx, k):
        _, ind = self.model.kneighbors(self.features[[idx]], n_neighbors=k + 1)
        ind = ind.ravel()
        return ind[ind != idx][:k]

    def _popularity_recs(self, idx, k):
        top = self.popular_order[: k + 1]
        return top[top != idx][:k]

    def _random_recs(self, idx, k):
        pick = self._rng.choice(len(self.data), size=k + 1, replace=False)
        return pick[pick != idx][:k]

    def _diversity(self, recs):
        k = len(recs)
        if k < 2:
            return None
        sim = cosine_similarity(self.features[recs])
        np.fill_diagonal(sim, 0)
        return 1 - sim.sum() / (k * (k - 1))

    def _score(self, recommend_fn, sample_idx, k):
        precisions, diversities, recommended = [], [], set()
        for idx in sample_idx:
            recs = recommend_fn(idx, k)
            recommended.update(recs.tolist())

            seed_genre = self.genres[idx]
            if seed_genre:  # skip queries with no genre label
                precisions.append(float(np.mean(self.genres[recs] == seed_genre)))

            d = self._diversity(recs)
            if d is not None:
                diversities.append(d)

        p = np.array(precisions)
        return {
            "precision_at_k": round(float(p.mean()), 4),
            "precision_ci95": round(float(1.96 * p.std(ddof=1) / np.sqrt(len(p))), 4),
            "diversity_score": round(float(np.mean(diversities)), 4),
            "catalog_coverage": round(len(recommended) / len(self.data), 4),
            "queries_scored": len(p),
        }

    def run_evaluation(self):
        self.load_artifacts()
        k = self.config.n_recommendations
        n = min(self.config.sample_size, len(self.data))
        sample_idx = self.data.sample(n, random_state=42).index.to_numpy()

        self._rng = np.random.default_rng(42)
        metrics = {
            "model": type(self.model).__name__,
            "k": k,
            "sample_size": n,
            "coverage_ceiling": round(min(1.0, n * k / len(self.data)), 4),
            "model_scores": self._score(self._model_recs, sample_idx, k),
            "popularity_baseline": self._score(self._popularity_recs, sample_idx, k),
            "random_baseline": self._score(self._random_recs, sample_idx, k),
        }

        with open(self.config.metrics_file, "w") as f:
            json.dump(metrics, f, indent=4)

        logger.info(f"Evaluation complete: {metrics}")
        return metrics