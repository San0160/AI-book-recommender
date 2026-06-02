from Book_recommender.pipeline.stage_01_DI import DataInjectionTrainingPipeline
from Book_recommender.pipeline.stage_02_DV import DataValidationTrainingPipeline
from Book_recommender.pipeline.stage_03_DP import DataProcessingTrainingPipeline
from Book_recommender.pipeline.stage_04_DT import DataTransformationTrainingPipeline
from Book_recommender.pipeline.stage_05_MT import ModelTrainerTrainingPipeline
from Book_recommender.pipeline.stage_06_ME import ModelEvaulationTrainingPipeline
from Book_recommender.logging.log import logger


STAGE_NAME = "Data Ingestion Stage"
try:
    logger.info(f">>>>>> Stage {STAGE_NAME} Started <<<<<<<<<<")
    data_injestion = DataInjectionTrainingPipeline()
    data_injestion.main()
    logger.info(f">>>>>>>>> Stage {STAGE_NAME} Completed <<<<<<<<<<")

except Exception as e:
    logger.exception(e)
    raise e

STAGE_NAME = "Data Validation Stage"
try:
    logger.info(f">>>>>> Stage {STAGE_NAME} Started <<<<<<<<<<")
    data_validation = DataValidationTrainingPipeline()
    data_validation.main()
    logger.info(f">>>>>>>>> Stage {STAGE_NAME} Completed <<<<<<<<<<")

except Exception as e:
    logger.exception(e)
    raise e

STAGE_NAME = "Data Processing Stage"
try:
    logger.info(f">>>>>> Stage {STAGE_NAME} Started <<<<<<<<<<")
    data_processing = DataProcessingTrainingPipeline()
    data_processing.main()
    logger.info(f">>>>>>>>> Stage {STAGE_NAME} Completed <<<<<<<<<<")

except Exception as e:
    logger.exception(e)
    raise e

STAGE_NAME = "Data Transformation Stage"
try:
    logger.info(f">>>>>> Stage {STAGE_NAME} Started <<<<<<<<<<")
    data_transformation = DataTransformationTrainingPipeline()
    data_transformation.main()
    logger.info(f">>>>>>>>> Stage {STAGE_NAME} Completed <<<<<<<<<<")

except Exception as e:
    logger.exception(e)
    raise e

STAGE_NAME = "Model Training Stage"
try:
    logger.info(f">>>>>> Stage {STAGE_NAME} Started <<<<<<<<<<")
    model_trainer = ModelTrainerTrainingPipeline()
    model_trainer.main()
    logger.info(f">>>>>>>>> Stage {STAGE_NAME} Completed <<<<<<<<<<")

except Exception as e:
    logger.exception(e)
    raise e

STAGE_NAME = "Model Evaluation Stage"
try:
    logger.info(f">>>>>> Stage {STAGE_NAME} Started <<<<<<<<<<")
    model_evaulation = ModelEvaulationTrainingPipeline()
    model_evaulation.main()
    logger.info(f">>>>>>>>> Stage {STAGE_NAME} Completed <<<<<<<<<<")

except Exception as e:
    logger.exception(e)
    raise e