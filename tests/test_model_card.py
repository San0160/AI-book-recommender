import json
from types import SimpleNamespace

import joblib
import scipy.sparse as sp

from Book_recommender.components.model_card import write_model_card


def test_card_has_what_the_page_needs(tmp_path):
    joblib.dump(sp.random(20, 7, density=0.5, format="csr", random_state=0), tmp_path / "f.pkl")
    cfg = SimpleNamespace(feature_matrix_path=tmp_path / "f.pkl", metrics_file=tmp_path / "metrics.json")
    params = {
        "model_trainer": {"model_name": "knn", "knn": {"n_neighbors": 10}},
        "data_transformation": {"method": "tfidf", "tfidf": {"min_df": 2}},
        "data_processing": {"feature_fields": ["title"]},
    }
    card = json.loads(write_model_card(cfg, params, {"model": "NearestNeighbors"}).read_text())
    assert card["books"] == 20 and card["model"]["n_features"] == 7
    assert card["model"]["feature_fields"] == ["title"]