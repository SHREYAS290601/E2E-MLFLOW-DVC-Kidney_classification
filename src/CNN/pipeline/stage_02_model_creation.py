from src.CNN.config.configuration import PretrainedModlelManager
from src.CNN.components.model_loader import PrepareBaseModel
from src.CNN import logger

STAGE_NAME = "Prepare Base Model"


class PrepareBaseModelPipeline:
    def __init__(self):
        pass

    def main(self):
        config = PretrainedModlelManager()
        prepare_base_model_config = config.prepare_base_model_config()
        prepare_base_model = PrepareBaseModel(config=prepare_base_model_config)
        prepare_base_model.get_base_model()
        prepare_base_model.updated_base_model()


if __name__ == "__main__":
    try:
        logger.info("********************************************************")
        logger.info(f"\n\n>>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = PrepareBaseModelPipeline()
        obj.main()
        logger.info(f"\n\n>>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e
