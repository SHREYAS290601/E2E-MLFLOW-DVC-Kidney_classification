from src.CNN.constants import *
import os
from src.CNN.utils.common import read_yaml, create_directories, save_json
from src.CNN.entity.config_entity import DataIngestionConfig, PrepareBaseModelConfig,TrainConfigs


class ConfigManager:
    def __init__(
        self, config_filepath=CONFIG_FILE_PATH, params_filepath=PARAMS_FILE_PATH
    ):
        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)

        create_directories([self.config.artifacts_roots])

    def get_data_ingestion_config(self) -> DataIngestionConfig:
        config = self.config.data_ingestion
        print(config)
        create_directories([config.root_dir])

        return DataIngestionConfig(
            root_dir=config.root_dir,
            source_URL=config.source_URL,
            local_data_file=config.local_data_file,
            unzip_dir=config.unzip_dir,
        )


class PretrainedModlelManager:
    def __init__(
        self, config_filepath=CONFIG_FILE_PATH, params_filepath=PARAMS_FILE_PATH
    ):
        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)

        create_directories([self.config.artifacts_roots])

    def prepare_base_model_config(self) -> PrepareBaseModelConfig:
        config = self.config.prepare_base_model
        # print(config)
        create_directories([config.root_dir])

        return PrepareBaseModelConfig(
            root_dir=Path(config.root_dir),
            base_model_path=Path(config.base_model_path),
            updated_model_path=Path(config.updated_base_model_path),
            params_image_size=self.params.IMAGE_SIZE,
            params_learning_rate=self.params.LEARNING_RATE,
            params_include_tops=self.params.INCLUDE_TOP,
            params_classes=self.params.CLASSES,
            params_weights=self.params.WEIGHTS,
        )


class ModelConfig:
    def __init__(
        self, config_filepath=CONFIG_FILE_PATH, params_filepath=PARAMS_FILE_PATH
    ):
        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)

        create_directories([self.config.artifacts_roots])

    def prepare_data_for_model(self) -> TrainConfigs:
        config = self.config.model_train

        return TrainConfigs(
            config.model_file_path,
            config.data_file_path,
            config.train_history,
            config.save_weights_path,
            self.params.IMAGE_SIZE,
            self.params.BATCH_SIZE,
            self.params.EPOCHS,
            self.params.CLASSES,
            self.params.featurewise_center,
            self.params.featurewise_std_normalization,
            self.params.rotation_range,
            self.params.width_shift_range,
            self.params.height_shift_range,
            self.params.horizontal_flip,
            self.params.validation_split,
            self.params.rescale,
        )
