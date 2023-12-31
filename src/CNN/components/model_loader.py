import urllib.request as request
from zipfile import ZipFile
import tensorflow as tf
from src.CNN.utils.common import read_yaml, create_directories, save_json
from src.CNN.entity.config_entity import PrepareBaseModelConfig
from src.CNN.constants import *
from pathlib import Path


class PrepareBaseModel:
    def __init__(self, config: PrepareBaseModelConfig):
        self.config = config

    def get_base_model(self):
        self.model = tf.keras.applications.MobileNetV2(
            input_shape=self.config.params_image_size,
            include_top=self.config.params_include_tops,
            weights=self.config.params_weights,
        )
        self.save_model(path=self.config.base_model_path, model=self.model)

    @staticmethod
    def save_model(path: Path, model: tf.keras.Model):
        model.save(path)

    @staticmethod
    def design_model(
        model, classes, freeze_all, learning_rate, freeze_start=None, freeze_end=None
    ):
        if freeze_all:
            for _ in model.layers:
                model.trainable = False
        elif (freeze_start is not None) and (
            (freeze_end > 0) and (freeze_end > freeze_start)
        ):
            for _ in model.layers[freeze_start:freeze_end]:
                model.trainable = False
        flatten_in = tf.keras.layers.Flatten()(model.output)
        prediction = tf.keras.layers.Dense((classes), activation="softmax")(flatten_in)

        full_model = tf.keras.models.Model(inputs=model.input, outputs=prediction)

        full_model.compile(
            loss="categorical_crossentropy",
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            metrics=["accuracy"],
        )
        full_model.summary()
        return full_model

    def updated_base_model(self):
        self.full_model = self.design_model(
            model=self.model,
            classes=self.config.params_classes,
            freeze_all=True,
            learning_rate=self.config.params_learning_rate,
        )
        self.save_model(path=self.config.updated_model_path, model=self.full_model)
