from src.CNN.config.configuration import ModelConfig
from src.CNN.components.model_train import TrainModel
import os
from src.CNN import logger
import mlflow


class TrainModelPipeline:
    def __init__(self):
        pass

    def main(self):
        model_config = ModelConfig()
        train_config = model_config.prepare_data_for_model()
        train_model = TrainModel(train_config)
        train_model.train()


if __name__ == "__main__":
    try:
        mlflow.set_tracking_uri(
            uri="https://dagshub.com/SHREYAS290601/E2E-MLFLOW-DVC-Kidney_classification.mlflow"
        )
        mlflow.set_experiment("Kidney_classification_train")
        logger.info("********************************************************")
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = TrainModelPipeline()
        obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e
