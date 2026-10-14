import pandas as pd

from Book_recommender.config.configuration import configurationManager
from Book_recommender.components.data_transformation import DataTransformation


class DataTransformationTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        config = configurationManager()
        transformation_config = config.get_data_transformation_config()

        data = pd.read_csv(transformation_config.data_path)

        transformation = DataTransformation(config=transformation_config)
        matrix = transformation.fit_transform(data)
        transformation.save(matrix)


if __name__ == "__main__":
    DataTransformationTrainingPipeline().main()