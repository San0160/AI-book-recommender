import pandas as pd
import joblib

from Book_recommender.config.configuration import configurationManager
from Book_recommender.components.model_trainer import ModelTrainer


class ModelTrainerTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        config = configurationManager()
        model_trainer_config = config.get_model_trainer_config()

        # Load processed data
        data = pd.read_csv(model_trainer_config.data_path)

        # Load saved TF-IDF vectorizer
        vectorizer = joblib.load(model_trainer_config.vectorizer_path)

        # Create TF-IDF matrix
        tfidf_matrix = vectorizer.transform(data["combined_features"])

        # Build, fit and save the model selected in params.yaml
        trainer = ModelTrainer(config=model_trainer_config)
        trainer.initialize_model()
        trainer.fit_model(tfidf_matrix)
        trainer.save()


if __name__ == "__main__":
    ModelTrainerTrainingPipeline().main()