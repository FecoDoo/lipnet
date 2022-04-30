import os
import numpy as np
from typing import List
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(".env")

from common.decode import create_decoder
from common.iters import chunks
from core.utils.video import (
    video_read,
    video_normalize,
)
from core.utils.config import LipNetConfig
from core.models.lipnet import LipNet
from utils.converter import roi_of_all_frames
from utils.logger import get_logger
from utils.helper import display_results


# define paths
ROOT = Path(__file__).parent.resolve()
DICTIONARY_PATH = ROOT.joinpath(os.environ["DICTIONARY_PATH"])
VIDEO_PATH = ROOT.joinpath(os.environ["VIDEO_PATH"])
LIPNET_WEIGHT_PATH = ROOT.joinpath(os.environ["LIPNET_WEIGHT_PATH"])
DENSENET_WEIGHT_PATH = ROOT.joinpath(os.environ["DENSENET_WEIGHT_PATH"])

logger = get_logger("lipnet")


def main():
    """
    Entry point of the script for using a trained model for predicting videos.
    """

    lipnet_weight_path = LIPNET_WEIGHT_PATH
    video_path = VIDEO_PATH

    if not lipnet_weight_path.is_file() or lipnet_weight_path.suffix != ".h5":
        logger.error("Trained weights path is not a valid file")
        return

    if not video_path.is_file() and not video_path.is_dir():
        logger.error("Path does not point to a video file nor to a directory")
        return

    # define params
    lipnet_config = LipNetConfig(
        int(os.environ["FRAME_COUNT"]),
        int(os.environ["IMAGE_WIDTH"]),
        int(os.environ["IMAGE_HEIGHT"]),
        int(os.environ["IMAGE_CHANNELS"]),
        int(os.environ["MAX_STRING"]),
    )
    predict(lipnet_config)


def predict(lipnet_config: LipNetConfig):
    logger.info("Loading weights at: {}".format(lipnet_config.model_weight_path))
    logger.info("Making predictions...")

    lipnet = LipNet(lipnet_config)
    lipnet.compile()
    lipnet.load_weights(lipnet_config.model_weight_path)

    valid_paths = []
    input_lengths = []
    predictions = None

    elapsed_videos = 0
    video_paths = get_list_of_videos(lipnet_config.video_path)

    for paths, lengths, y_pred in predict_batches(lipnet, video_paths):
        valid_paths += paths
        input_lengths += lengths

        predictions = (
            y_pred if predictions is None else np.append(predictions, y_pred, axis=0)
        )

        y_pred_len = len(y_pred)
        elapsed_videos += y_pred_len

        logger.info(
            "Predicted batch of {} videos\t({} elapsed)".format(
                y_pred_len, elapsed_videos
            )
        )

    decoder = create_decoder(DICTIONARY_PATH)
    results = decode_predictions(predictions, input_lengths, decoder)

    logger.info("RESULTS:")

    display = True
    visualize = False

    if display or visualize:
        display_results(valid_paths, results, display, visualize)


def get_list_of_videos(path: os.PathLike) -> List[str]:
    if path.is_file():
        logger.info("Predicting for video at: {}".format(path))
        video_paths = [path]
    else:
        logger.info("Predicting batch at: {}".format(path))
        video_paths = get_video_files_in_dir(path)

    return video_paths


def get_video_files_in_dir(path: str) -> List[str]:
    return [f for ext in ["*.mpg", "*.npy"] for f in path.glob(ext)]


def get_video_data(path: str) -> np.ndarray:
    video = video_read(path)
    data = roi_of_all_frames(video)
    return video_normalize(data) if data is not None else None


def predict_batches(lipnet: LipNet, video_paths: List[str]):
    batch_size = int(os.environ["BATCH_SIZE"])

    for paths in chunks(video_paths, batch_size):
        input_data = [(p, get_video_data(p)) for p in paths]
        input_data = [x for x in input_data if x[1] is not None]

        if len(input_data) <= 0:
            continue

        valid_paths = [x[0] for x in input_data]

        x_data = np.array([x[1] for x in input_data])
        lengths = [len(x) for x in x_data]

        y_pred = lipnet.predict(x_data)

        yield (valid_paths, lengths, y_pred)


def decode_predictions(y_pred: np.ndarray, input_lengths: list, decoder) -> list:
    input_lengths = np.array(input_lengths)
    return decoder.decode(y_pred, input_lengths)


if __name__ == "__main__":
    main()
