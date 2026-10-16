import json
from datetime import datetime, timezone
from pathlib import Path

import joblib

from Book_recommender.logging.log import logger

CARD_NAME = "model_card.json"


def build_model_card(eval_cfg, params, metrics) -> dict:
    features = joblib.load(eval_cfg.feature_matrix_path)
    model_key = params["model_trainer"]["model_name"]
    method = params["data_transformation"]["method"]

    return {
        "schema": 1,
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "books": int(features.shape[0]),
        "model": {
            "algorithm": metrics["model"],
            "model_name": model_key,
            "params": dict(params["model_trainer"][model_key]),
            "feature_method": method,
            "feature_params": dict(params["data_transformation"][method]),
            "feature_fields": list(params["data_processing"]["feature_fields"]),
            "n_features": int(features.shape[1]),
        },
        "gate": dict(params.get("quality_gate") or {}),
        "evaluation": metrics,
    }


def write_model_card(eval_cfg, params, metrics) -> Path:
    path = Path(eval_cfg.metrics_file).with_name(CARD_NAME)
    path.write_text(json.dumps(build_model_card(eval_cfg, params, metrics), indent=2), encoding="utf-8")
    logger.info(f"Model card written to {path}")
    return path