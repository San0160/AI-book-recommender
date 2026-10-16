from Book_recommender.config.configuration import configurationManager
from Book_recommender.components.model_evaluation import ModelEvaluation
from Book_recommender.components.experiment_tracker import log_run
from Book_recommender.components.model_card import write_model_card
from Book_recommender.logging.log import logger


class ModelEvaulationTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        config = configurationManager()
        evaluation_config = config.get_model_evaluation_config()

        metrics = ModelEvaluation(config=evaluation_config).run_evaluation()
        print(metrics)

        try:
            log_run(evaluation_config, metrics, config.params)
        except Exception:
            # Tracking is bookkeeping; it should never fail the pipeline.
            logger.exception("MLflow logging failed; metrics.json was still saved")

        metrics = ModelEvaluation(config=evaluation_config).run_evaluation()
        print(metrics)
        write_model_card(evaluation_config, config.params, metrics)


if __name__ == "__main__":
    ModelEvaulationTrainingPipeline().main()