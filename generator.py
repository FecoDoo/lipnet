import json
import os
import logging
from pathlib import Path
from multiprocessing import Pool, log_to_stderr
from utils.validator import validate_preprocessing_config
from pathlib import Path
from common.files import is_dir
from preprocessing.extract import extract

# define logger
logger = log_to_stderr()
logger.setLevel(level=logging.DEBUG)

handler = logging.StreamHandler()
logger.addHandler(handler)


@validate_preprocessing_config
def generator(root_path: os.PathLike, config_path: os.PathLike):
    """
    extract features from *.mpg files and convert into *.npy format
    """

    with open(config_path, "r") as c:
        config = json.load(c)["preprocessing"]

    dataset_path = root_path.joinpath(config["dataset_path"]).resolve()
    del config["dataset_path"]

    groups = list(dataset_path.glob("*"))

    for i in groups:
        if not is_dir(i):
            logger.error(f"group {i} is not a directory")
            groups.remove(i)
            continue

    with Pool(processes=None) as pool:
        res = [
            pool.apply_async(
                extract,
                args=(
                    group_path,
                    root_path,
                    config,
                    logger,
                ),
            )
            for group_path in groups
        ]

        for p in res:
            p.get()


if __name__ == "__main__":
    # try:
    root_path = Path(os.path.dirname(__file__))

    config_path = root_path.joinpath("config/config.json")

    generator(root_path=root_path, config_path=config_path)

    # except Exception as e:
    #     print(e)
