from src.CNN.entity.config_entity import *
from src.CNN.utils.common import *
import mlflow
import mlflow.keras
from tensorflow.keras.models import load_model
from keras.utils import to_categorical
from glob import glob
import cv2
import json
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from src.CNN import logger
from src.CNN.entity.config_entity import EvalConfig
import time
import mlflow.keras
from urllib.parse import urlparse
from pathlib import Path

os.environ["KERAS_BACKEND"] = "tensorflow"
import keras


class EvaluateModel:
    def __init__(self, config: EvalConfig):
        self.config = config
        self.data = []
        self.class_name = []

    def evaluate(self):
        self.get_eval_data(self)
        evaldata = self.create_eval_data(self)
        self.model = load_model("./artifacts/model_train/model_weights/weights.h5")
        logger.info(f"Model is loaded from {self.config.path_to_model}")
        self.score = self.model.evaluate(evaldata)
        print("Loss: ", self.score[0])
        print("Accuracy: ", self.score[1])
        logger.info(f"Loss: {self.score[0]}")
        logger.info(f"Accuracy: {self.score[1]}")
        logger.info(f"Accuracy is {self.score[1]*100}%")

    def log_into_mlflow(self):
        mlflow.set_registry_uri(self.config.mlflow_uri)
        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

        with mlflow.start_run():
            mlflow.log_params(self.config.all_params)
            mlflow.log_metrics({"loss": self.score[0], "accuracy": self.score[1]})
            # Model registry does not work with file store
            if tracking_url_type_store != "file":
                mlflow.keras.log_model(
                    self.model, "model", registered_model_name="MobileNetV2"
                )
                mlflow.set_tag("mlflow.runName", f"MobileNetV2_Eval_{time.time()}")
            else:
                mlflow.keras.log_model(self.model, "model")

    @staticmethod
    def get_eval_data(self):
        for index, folder in enumerate(glob(f"{self.config.training_data}/*")):
            for _, image in enumerate(glob(f"{folder}/*.jpg")):
                if _ <= 20:
                    img = cv2.imread(image)
                    img = cv2.resize(
                        img,
                        tuple(self.config.params_image_size[:-1]),
                        interpolation=cv2.INTER_AREA,
                    )
                    img = np.array(img)
                    self.data.append(img)
                    self.class_name.append(index)
        print("Data Loaded")
        logger.info("Data Loaded")

    @staticmethod
    def create_eval_data(self):
        evalgen = ImageDataGenerator(
            featurewise_center=self.config.all_params.featurewise_center,
            featurewise_std_normalization=self.config.all_params.featurewise_std_normalization,
            rotation_range=self.config.all_params.rotation_range,
            width_shift_range=self.config.all_params.width_shift_range,
            height_shift_range=self.config.all_params.height_shift_range,
            horizontal_flip=self.config.all_params.horizontal_flip,
            rescale=self.config.all_params.rescale,
        )
        evalgen.fit(self.data)
        self.data = np.array(self.data)
        self.class_name = np.array(
            to_categorical(self.class_name, num_classes=self.config.all_params.CLASSES)
        )
        return evalgen.flow(self.data, self.class_name)
