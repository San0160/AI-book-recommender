"""Fail the build if the freshly trained model is worse than the agreed floor."""
import json
import sys

from Book_recommender.config.configuration import configurationManager


def main() -> int:
    cm = configurationManager()
    gate = cm.params.quality_gate
    with open(cm.get_model_evaluation_config().metrics_file) as f:
        metrics = json.load(f)

    model_p = metrics["model_scores"]["precision_at_k"]
    random_p = metrics["random_baseline"]["precision_at_k"]
    print(f"precision@k={model_p}  random={random_p}  ({metrics['model']})")

    failures = []
    if model_p < gate.min_precision_at_k:
        failures.append(f"precision {model_p} < floor {gate.min_precision_at_k}")
    if model_p < gate.min_lift_over_random * random_p:
        failures.append(f"precision {model_p} < {gate.min_lift_over_random}x random ({random_p})")

    for msg in failures:
        print("QUALITY GATE FAILED:", msg)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())