import mlflow

from Book_recommender.logging.log import logger


def _flatten(d, prefix=""):
    """{'a': {'b': 1}} -> {'a.b': 1}"""
    out = {}
    for key, value in d.items():
        name = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            out.update(_flatten(value, name))
        else:
            out[name] = value
    return out


def log_run(config, metrics, params):
    """Record one pipeline run: params, metrics and artifacts."""
    active_model = params["model_trainer"]["model_name"]

    # Log only the settings that actually shaped this run,
    # not the unused model blocks.
    used_params = {
        "data_processing": params["data_processing"],
        "data_transformation": params["data_transformation"],
        "model_trainer": {
            "model_name": active_model,
            active_model: params["model_trainer"][active_model],
        },
    }

    mlflow.set_tracking_uri(config.mlflow_tracking_uri)
    mlflow.set_experiment(config.mlflow_experiment_name)

    with mlflow.start_run(run_name=active_model):
        mlflow.log_params({k: str(v) for k, v in _flatten(used_params).items()})
        mlflow.log_params({"k": metrics["k"], "sample_size": metrics["sample_size"]})

        scores = metrics["model_scores"]
        mlflow.log_metrics({
            "precision_at_k": scores["precision_at_k"],
            "precision_ci95": scores["precision_ci95"],
            "diversity_score": scores["diversity_score"],
            "catalog_coverage": scores["catalog_coverage"],
            "popularity_precision_at_k": metrics["popularity_baseline"]["precision_at_k"],
            "random_precision_at_k": metrics["random_baseline"]["precision_at_k"],
        })

        mlflow.set_tag("model_class", metrics["model"])
        mlflow.log_artifact(str(config.metrics_file))
        mlflow.log_artifact("params.yaml")

    logger.info("Run logged to MLflow")