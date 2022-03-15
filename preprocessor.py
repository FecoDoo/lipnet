from inspect import trace
import json
import os
import logging
import traceback
import env
from multiprocessing import Pool, log_to_stderr, Lock
from utils.validator import validate_preprocessing_config
from pathlib import Path
from utils.extract import extract
from colorama import Fore, init, deinit

INFO_LOG = Fore.BLUE
ERROR_LOG = Fore.RED
DEBUG_LOG = Fore.YELLOW

flag = "dev"


@validate_preprocessing_config
def generator(root_path: os.PathLike, config_path: os.PathLike):
    """
    extract features from *.mpg files and convert into *.npy format
    """

    logger.info(
        r"""
   __         __     ______   __   __     ______     __  __     ______  
  /\ \       /\ \   /\  == \ /\ "-.\ \   /\  ___\   /\_\_\_\   /\__  _\ 
  \ \ \____  \ \ \  \ \  _-/ \ \ \-.  \  \ \  __\   \/_/\_\/_  \/_/\ \/ 
   \ \_____\  \ \_\  \ \_\    \ \_\\"\_\  \ \_____\   /\_\/\_\    \ \_\ 
    \/_____/   \/_/   \/_/     \/_/ \/_/   \/_____/   \/_/\/_/     \/_/ 
	"""
    )

    try:
        with open(config_path, "r") as c:
            config = json.load(c)["preprocessing"]

        dataset_path = root_path.joinpath(config["dataset_path"]).resolve()
        del config["dataset_path"]

        groups = list(dataset_path.glob("*"))

        for i in groups:
            if not i.is_dir():
                logger.error(f"{ERROR_LOG} group {i} is not a directory")
                groups.remove(i)
                continue

    except Exception:
        print(traceback.format_exc())

    if env.DEV:
        # dev
        extract(groups[0], root_path, config, logger)
    else:
        lock = Lock()
        with Pool(processes=None) as pool:
            res = [
                pool.apply_async(
                    extract, args=(group_path, root_path, config, logger, lock,),
                )
                for group_path in groups
            ]

            for p in res:
                p.get()


if __name__ == "__main__":
    # define logger
    logger = log_to_stderr(level=logging.DEBUG)

    init(autoreset=True)

    root_path = Path(__file__).parents[0]
    config_path = root_path.joinpath("config/config.json")

    generator(root_path=root_path, config_path=config_path)

    deinit()
