import os
from box.exceptions import BoxValueError
import yaml
from src.CNN import logger
import json, joblib, base64
from ensure import ensure_annotations
from pathlib import Path
from box import ConfigBox
from typing import Dict, List, Optional, Union, Tuple, Any


@ensure_annotations
def read_yaml(path_to_yaml: Path) -> ConfigBox:
    """Reads a yaml file and returns a ConfigBox object

    Args:
        path_to_yaml (Path): path to yaml file

    Returns:
        ConfigBox: ConfigBox object

    Raises:
        ValueError: if the file is empty
        e:empty file

    """
    try:
        with open(path_to_yaml, "r") as yaml_file:
            content = yaml.safe_load(yaml_file)
            logger.info(f"Yaml File : {path_to_yaml} Loaded Successfully")
            return ConfigBox(content)
    except BoxValueError as e:
        logger.error(f"Empty File : {path_to_yaml}")
    except Exception as e:
        logger.error(e)
        raise e


@ensure_annotations
def create_directories(path_to_directories: list, verbose=True):
    """Create list of directories

    Args:
        path_to_directories (list): list of oath of directories
        verbose (bool, optional): _description_. Defaults to True.
    """
    try:
        for path in path_to_directories:
            os.makedirs(path, exist_ok=True)
            if verbose:
                logger.info(f"Created Directory : {path}")
    except Exception as e:
        logger.error(e)


@ensure_annotations
def save_json(path: Path, data: dict):
    """Save json file

    Args:
        path (Path): path to json file
        data (dict): data to be saved
    """
    try:
        with open(path, "w") as f:
            json.dump(data, f, indent=4)
            logger.info(f"Saved json file : {path}")
    except Exception as e:
        logger.error(e)


@ensure_annotations
def load_json(path: Path) -> ConfigBox:
    """load json files data

    Args:
        path (Path): path to json file

    Returns:
        ConfigBox: data as class attributes instead of dict
    """
    with open(path) as f:
        content = json.load(f)

    logger.info(f"json file loaded succesfully from: {path}")
    return ConfigBox(content)


@ensure_annotations
def save_bin(data: Any, path: Path):
    """save binary file

    Args:
        data (Any): data to be saved as binary
        path (Path): path to binary file
    """
    joblib.dump(value=data, filename=path)
    logger.info(f"binary file saved at: {path}")


@ensure_annotations
def load_bin(path: Path) -> Any:
    """load binary data

    Args:
        path (Path): path to binary file

    Returns:
        Any: object stored in the file
    """
    data = joblib.load(path)
    logger.info(f"binary file loaded from: {path}")
    return data


@ensure_annotations
def get_size(path: Path) -> str:
    """get size in KB

    Args:
        path (Path): path of the file

    Returns:
        str: size in KB
    """
    size_in_kb = round(os.path.getsize(path) / 1024)
    return f"~ {size_in_kb} KB"


def decodeImage(imgstring, fileName):
    imgdata = base64.b64decode(imgstring)
    with open(fileName, "wb") as f:
        f.write(imgdata)
        f.close()


def encodeImageIntoBase64(croppedImagePath):
    with open(croppedImagePath, "rb") as f:
        return base64.b64encode(f.read())
