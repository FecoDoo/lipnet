import json
import os

from importlib_metadata import pathlib
from common.files import get_file_extension, is_dir, is_file
from jsonschema import validate
from functools import wraps
from pathlib import Path


def validate_preprocessing_config(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        """
        validating default config file
        """
        root_path = kwargs.get("root_path", None)
        config_path = kwargs.get("config_path", None)

        if not is_file(config_path):
            raise FileNotFoundError("Invalid path to config file")

        schema = {
            "type": "object",
            "properties": {
                "dataset_path": {"type": "string"},
                "output_path": {"type": "string"},
                "predictor_path": {"type": "string"},
                "log_path": {"type": "string"},
                "pattern": {"type": "string"},
            },
        }

        with open(config_path, "r") as c:
            config = json.load(c)["preprocessing"]

        validate(instance=config, schema=schema)

        if not is_dir(root_path.joinpath(config["dataset_path"]).resolve()):
            raise NotADirectoryError(
                f"Invalid path to dataset directory: {config['dataset_path']}"
            )

        if not is_dir(root_path.joinpath(config["output_path"]).resolve()):
            raise NotADirectoryError(
                f"Invalid path to output directory: {config['output_path']}"
            )

        if not is_file(root_path.joinpath(config["predictor_path"]).resolve()):
            raise FileNotFoundError(
                f"Invalid path to predictor file: {config['predictor_path']}"
            )

        if (
            get_file_extension(root_path.joinpath(config["predictor_path"]).resolve())
            != ".dat"
        ):
            raise ValueError("Invalid predictor file type")

        if not is_dir(root_path.joinpath(config["log_path"]).resolve()):
            raise FileNotFoundError("Invalid path to logs directory")

        return func(*args, **kwargs)

    return wrapper
