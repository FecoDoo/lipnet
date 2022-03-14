import json
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

        if not root_path:
            raise ValueError("Missing root_path")

        if not config_path:
            raise ValueError("Missing config_path")

        if not Path(config_path).is_file():
            raise FileNotFoundError("Invalid path of config file")

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

        # dataset path
        if not root_path.joinpath(config["dataset_path"]).resolve().is_dir():
            raise NotADirectoryError(
                f"Invalid path to dataset directory: {config['dataset_path']}"
            )

        # output path
        output_path = root_path.joinpath(config["output_path"]).resolve()
        output_path.mkdir(exist_ok=True)

        # predictor
        if not root_path.joinpath(config["predictor_path"]).resolve().is_file():
            raise FileNotFoundError(
                f"Invalid path to predictor file: {config['predictor_path']}"
            )

        if not root_path.joinpath(config["cnn_predictor_path"]).resolve().is_file():
            raise FileNotFoundError(
                f"Invalid path to predictor file: {config['cnn_predictor_path']}"
            )

        if root_path.joinpath(config["predictor_path"]).resolve().suffix != ".dat":
            raise ValueError("Invalid predictor file type")

        # log path
        if not root_path.joinpath(config["log_path"]).resolve().is_dir():
            raise FileNotFoundError("Invalid path of log directory")

        return func(*args, **kwargs)

    return wrapper
