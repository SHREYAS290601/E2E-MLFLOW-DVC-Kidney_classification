from dataclasses import dataclass
from pathlib import Path
from src.CNN.constants import *
from src.CNN.utils.common import read_yaml, create_directories
import numpy as np


@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir: Path
    source_URL: str
    local_data_file: Path
    unzip_dir: Path


@dataclass(frozen=True)
class PrepareBaseModelConfig:
    root_dir: Path
    base_model_path: Path
    updated_model_path: Path
    params_image_size: list
    params_learning_rate: float
    params_include_tops: bool
    params_weights: str
    params_classes: int


@dataclass(frozen=True)
class TrainConfigs:
    model_file_path: Path
    data_file_path: Path
    train_history: str
    save_weights_path: Path
    IMAGE_SIZE: list
    BATCH_SIZE: int
    EPOCHS: int
    CLASSES: int
    featurewise_center: bool
    featurewise_std_normalization: bool
    rotation_range: int
    width_shift_range: float
    height_shift_range: float
    horizontal_flip: bool
    validation_split: float
    rescale: float

@dataclass(frozen=True)
class EvalConfig:
    path_to_model: Path
    training_data: Path
    all_params: dict
    mlflow_uri: str
    params_image_size: list
    params_batch: int