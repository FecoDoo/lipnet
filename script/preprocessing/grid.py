import os
import sys
import logging
from dotenv import load_dotenv
from multiprocessing import Pool, log_to_stderr, Lock
from pathlib import Path
from datetime import datetime

root = Path(__file__).parents[2].resolve()

sys.path.insert(0, str(root))

assert load_dotenv(".env")

timestamp = datetime.utcnow().strftime(os.environ["DATETIME_FMT"])

GRID_DATASET_PATH = root.joinpath("data/grid/videos")
GRID_ALIGN_PATH = root.joinpath("data/grid/aligns")
GRID_NPY_PATH = root.joinpath("data/grid/npy")
DICTIONARY_PATH = root.joinpath("data/dictionaries/grid.txt")
LOG_PATH = root.joinpath("logs", "lipnet", "preprocessing")


if not GRID_NPY_PATH.exists():
    GRID_NPY_PATH.mkdir()

if not LOG_PATH.exists():
    LOG_PATH.mkdir()

from core.utils.video import (
    video_to_numpy,
    video_read,
)

video_suffix = ".mpg"

# define logger
logger = log_to_stderr(level=logging.INFO)

lock = Lock()


def convert_videos(path: Path) -> None:

    group = path.name

    # create output dir for each subject
    output_dir = GRID_NPY_PATH.joinpath(group)
    output_dir.mkdir(exist_ok=True)

    # record corrupted videos
    videos_failed = []

    logger.info(f"Start: {group}")

    for video_path in path.glob("*" + video_suffix):
        try:
            # get identifier
            hash_string = video_path.stem
            output_path = output_dir.joinpath(hash_string + ".npy")

            # skip existing outputs
            if output_path.is_file():
                continue

            stream = video_read(video_path, num_frames=75)

            if stream is None or stream.shape[0] != 75:
                raise ValueError(f"{video_path} has wrong shape")

            video_to_numpy(stream=stream, output_path=output_path)

        except Exception as e:
            logger.error(e.args)
            videos_failed.append(str(video_path) + "\n")
            continue

    with lock:
        with open(LOG_PATH.joinpath("corrupted_videos.txt"), "a") as fp:
            fp.writelines(videos_failed)

    logger.info(f"{path.name} processing completed.")


if __name__ == "__main__":
    groups = list(GRID_DATASET_PATH.iterdir())

    for i in groups:
        if not i.is_dir():
            logger.error(f"Group {i} is not a directory")
            groups.remove(i)
            continue

    with Pool(processes=None) as pool:
        res = [
            pool.apply_async(
                convert_videos,
                args=(group_path,),
            )
            for group_path in groups
        ]

        for p in res:
            p.get()
