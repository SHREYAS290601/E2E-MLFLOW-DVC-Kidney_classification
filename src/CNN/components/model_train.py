from tensorflow.keras.models import load_model
from keras.utils import to_categorical
from glob import glob
import cv2
import json
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from src.CNN import logger
from src.CNN.entity.config_entity import TrainConfigs
import numpy as np


class TrainModel:
    def __init__(self, config: TrainConfigs):
        self.config = config
        self.data = []
        self.class_name = []

    def train(self):
        self.load_data(self, self.config.data_file_path)
        train, valid = self.preprocess_data(self)
        model = load_model("./artifacts/prepare_base_model/base_model_update.h5")
        logger.info(f"Model is loaded from {self.config.model_file_path}")
        history = model.fit_generator(
            train,
            epochs=self.config.EPOCHS,
            validation_data=(valid),
        )
        model.save(f"{self.config.save_weights_path}/weights.h5")
        with open(f"{self.config.train_history}", "w") as f:
            json.dump(history.history, f)
            logger.info(f"Saving history at {self.config.train_history}")
        logger.info(f"Model is saved at {self.config.model_file_path}")
        # print("Model is saved at {}".format(self.config.model_file_path))

    @staticmethod
    def preprocess_data(self, augmentation=True):
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(
            self.data, self.class_name, test_size=0.2, random_state=42
        )
        self.y_train = to_categorical(self.y_train, num_classes=self.config.CLASSES)
        self.y_test = to_categorical(self.y_test, num_classes=self.config.CLASSES)
        # print(self.x_train[0].shape)
        datagen = ImageDataGenerator(
            featurewise_center=self.config.featurewise_center,
            featurewise_std_normalization=self.config.featurewise_std_normalization,
            rotation_range=self.config.rotation_range,
            width_shift_range=self.config.width_shift_range,
            height_shift_range=self.config.height_shift_range,
            horizontal_flip=self.config.horizontal_flip,
            rescale=self.config.rescale,
        )
        validgen = ImageDataGenerator(
            featurewise_center=self.config.featurewise_center,
            featurewise_std_normalization=self.config.featurewise_std_normalization,
            rotation_range=self.config.rotation_range,
            width_shift_range=self.config.width_shift_range,
            height_shift_range=self.config.height_shift_range,
            horizontal_flip=self.config.horizontal_flip,
            rescale=self.config.rescale,
        )
        try:
            datagen.fit(self.x_train)
            validgen.fit(self.x_test)
            # print(dir(train_gen.flow))
            print("Data Augmentation Done")
            logger.info("Data Augmentation Done")
            self.x_train = np.array(self.x_train)
            self.x_test = np.array(self.x_test)
            self.y_train = np.array(self.y_train)
            self.y_test = np.array(self.y_test)
            train_data = datagen.flow(self.x_train, self.y_train)
            valid_data = validgen.flow(self.x_test, self.y_test)
            return train_data, valid_data
        except Exception as e:
            print(e)
            # logger.info(e)
        # print(x_test_processed[0])

    @staticmethod
    def load_data(self, data):
        for index, folder in enumerate(glob(f"{self.config.data_file_path}/*")):
            for image in glob(f"{folder}/*.jpg"):
                img = cv2.imread(image)
                img = cv2.resize(
                    img,
                    tuple(self.config.IMAGE_SIZE[:-1]),
                    interpolation=cv2.INTER_AREA,
                )
                img = np.array(img)
                self.data.append(img)
                self.class_name.append(index)
        print("Data Loaded")
        logger.info("Data Loaded")
