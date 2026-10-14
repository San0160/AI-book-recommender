import joblib

from Book_recommender.config.configuration import configurationManager
from Book_recommender.components.model_trainer import ModelTrainer


class ModelTrainerTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        config = configurationManager()
        model_trainer_config = config.get_model_trainer_config()

        features = joblib.load(model_trainer_config.feature_matrix_path)

        trainer = ModelTrainer(config=model_trainer_config)
        trainer.initialize_model()
        trainer.fit_model(features)
        trainer.save()


if __name__ == "__main__":
    ModelTrainerTrainingPipeline().main()