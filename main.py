from src.CNN.pipeline.stage_01_data_ingestion import (
    DataIngestionTrainingPipeline,
)
from src.CNN.pipeline.stage_02_model_creation import PrepareBaseModelPipeline
from src.CNN.pipeline.stage_03_train_model import TrainModelPipeline
from src.CNN.pipeline.stage_04_model_eval import EvalModelPipeline
from src.CNN import logger
import os

os.environ[
    "MLFLOW_TRACKING_URI"
] = "https://dagshub.com/SHREYAS290601/E2E-MLFLOW-DVC-Kidney_classification.mlflow"
os.environ["MLFLOW_TRACKING_USERNAME"] = "SHREYAS290601"
os.environ["MLFLOW_TRACKING_PASSWORD"] = "b06e74ec71a76270575d2e406cbc1d9f87f6a357"
STAGE_NAME = "Data Ingestion stage"
try:
    logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
    data_ingestion = DataIngestionTrainingPipeline()
    data_ingestion.main()
    logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\n")
except Exception as e:
    logger.exception(e)
    raise e

STAGE_NAME = "Prepare Base Model"
try:
    logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
    data_ingestion = PrepareBaseModelPipeline()
    data_ingestion.main()
    logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\n")
except Exception as e:
    logger.exception(e)
    raise e

STAGE_NAME = "Train Model"
try:
    logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
    data_ingestion = TrainModelPipeline()
    data_ingestion.main()
    logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\n")
except Exception as e:
    logger.exception(e)
    raise e

STAGE_NAME = "Evaluate Model"
try:
    logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
    data_ingestion = EvalModelPipeline()
    data_ingestion.main()
    logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\n")
except Exception as e:
    logger.exception(e)
    raise e
