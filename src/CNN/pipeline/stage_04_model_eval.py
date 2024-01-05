from src.CNN.config.configuration import EvalModelManager
from src.CNN.components.model_eval import EvaluateModel
import os
from src.CNN import logger
import mlflow


class EvalModelPipeline:
    def __init__(self):
        pass

    def main(self):
        model_config = EvalModelManager()
        eval_config = model_config.evaluate_model_config()
        eval_model = EvaluateModel(eval_config)
        eval_model.evaluate()
        eval_model.log_into_mlflow()


if __name__ == "__main__":
    try:
        mlflow.set_tracking_uri(
            uri="https://dagshub.com/SHREYAS290601/E2E-MLFLOW-DVC-Kidney_classification.mlflow"
        )
        mlflow.set_experiment("Kidney_classification_eval")
        logger.info("********************************************************")
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = EvalModelPipeline()
        obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e
