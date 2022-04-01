import os
import logging
import traceback
import env
from multiprocessing import Pool, log_to_stderr
from pathlib import Path
from utils.converter import Converter
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[0]
DATA_TARGET_DIR = ROOT.joinpath("data/dataset")
ERROR_DIR = ROOT.joinpath("logs/preprocessing")

if not DATA_TARGET_DIR.exists():
    DATA_TARGET_DIR.mkdir()

if not ERROR_DIR.exists():
    ERROR_DIR.mkdir()

pattern = env.VIDEO_PATTERN

# define logger
logger = log_to_stderr(level=logging.DEBUG)


def convert_video_to_array(group_path: os.PathLike):
    """
    convert features from *.mpg files and convert into *.npy format
    """

    try:
        groupname = group_path.name

        video_target_dir = DATA_TARGET_DIR.joinpath(groupname)
        video_target_dir.mkdir(exist_ok=True)

        videos_failed = []

        logger.info(f"Start: {groupname}")

        converter = Converter(logger)

        for file_path in group_path.glob(pattern):
            hash_string = file_path.stem
            output_path = video_target_dir.joinpath(hash_string + ".npy")

            if output_path.is_file():
                logger.info(f"{groupname + ' | ' + hash_string} | skipped")
                continue

            if not converter.video_to_frames(file_path, output_path):
                videos_failed.append(hash_string + "\n")

        with open(str(ERROR_DIR.joinpath(groupname + ".txt")), "w") as f:
            f.writelines(videos_failed)

    except Exception:
        logger.error(traceback.format_exc())
    finally:
        logger.info(f"{group_path.name} processing completed.")


# @validate_preprocessing_config
def manager():
    """
    Multiprocessing manager
    """

    try:
        data_source_dir = ROOT.joinpath("../dataset/lipnet/train/").resolve()
        data_target_dir = ROOT.joinpath("data/dataset").resolve()

        if not data_source_dir.is_dir():
            raise FileNotFoundError("data source directory not found")

        if not data_target_dir.is_dir():
            raise FileNotFoundError("data target directory not found")

        groups = list(data_source_dir.glob("*"))

        for i in groups:
            if not i.is_dir():
                logger.error(f"Group {i} is not a directory")
                groups.remove(i)
                continue

        with Pool(processes=None) as pool:
            res = [
                pool.apply_async(
                    convert_video_to_array,
                    args=(group_path,),
                )
                for group_path in groups
            ]

            for p in res:
                p.get()

    except Exception:
        logger.error(traceback.format_exc())


if __name__ == "__main__":
    manager()
