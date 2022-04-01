import argparse
import csv
import os
import numpy as np
import skvideo.io
import env
import logging

from typing import NamedTuple, List
from pathlib import Path
from common.decode import create_decoder
from common.iters import chunks
from core.helpers.video import (
    get_video_data_from_file,
    reshape_and_normalize_video_data,
)
from core.model.lipnet import LipNet
from core.utils.visualization import visualize_video_subtitle
from utils.converter import Converter


os.environ["TF_CPP_MIN_LOG_LEVEL"] = env.TF_CPP_MIN_LOG_LEVEL
logger = logging.getLogger("predicting")
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler())

# define paths
ROOT = Path(os.path.dirname(os.path.realpath(__file__))).resolve()
DICTIONARY_PATH = ROOT.joinpath(env.DICTIONARY_PATH)
VIDEO_PATH = ROOT.joinpath(env.VIDEO_PATH)
MODEL_PATH = ROOT.joinpath(env.MODEL_PATH)


class PredictConfig(NamedTuple):
    model: os.PathLike
    video_path: os.PathLike
    frame_count: int = env.FRAME_COUNT
    image_width: int = env.IMAGE_WIDTH
    image_height: int = env.IMAGE_HEIGHT
    image_channels: int = env.IMAGE_CHANNELS
    max_string: int = env.MAX_STRING


def main():
    """
    Entry point of the script for using a trained model for predicting videos.
    """

    model = MODEL_PATH
    video = VIDEO_PATH

    if not model.is_file() or model.suffix != ".h5":
        logger.error("\nERROR: Trained weights path is not a valid file")
        return

    if not video.is_file() and not video.is_dir():
        logger.error("\nERROR: Path does not point to a video file nor to a directory")
        return

    config = PredictConfig(model, video)
    predict(config)


def predict(config: PredictConfig):
    logger.info("Loading weights at: {}".format(config.model))

    logger.info("\nMaking predictions...\n")

    lipnet = LipNet(
        config.frame_count,
        config.image_channels,
        config.image_height,
        config.image_width,
        config.max_string,
    )
    lipnet.compile()
    lipnet.load_weights(config.model)

    valid_paths = []
    input_lengths = []
    predictions = None

    elapsed_videos = 0
    video_paths = get_list_of_videos(config.video_path)

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

    logger.info("\n\nRESULTS:\n")

    display = query_yes_no("List all prediction outputs?", True)
    visualize = query_yes_no("Visualize as video captions?", False)

    save_csv = query_yes_no("Save prediction outputs to a .csv file?", False)

    if save_csv:
        csv_path = query_save_csv_path()
        write_results_to_csv(csv_path, valid_paths, results)

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


def get_video_data(path: str, converter) -> np.ndarray:
    if path.suffix == ".mpg":
        data = converter.extract_video_data(path)
        return reshape_and_normalize_video_data(data) if data is not None else None
    else:
        return get_video_data_from_file(path)


def get_entire_video_data(path: str) -> np.ndarray:
    if path.suffix == ".mpg":
        return np.swapaxes(skvideo.io.vread(path), 1, 2)
    else:
        return get_video_data_from_file(path)


def predict_batches(lipnet: LipNet, video_paths: List[str]):
    batch_size = env.BATCH_SIZE

    converter = Converter(logger)

    for paths in chunks(video_paths, batch_size):
        input_data = [(p, get_video_data(p, converter)) for p in paths]
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


def query_yes_no(query: str, default: bool = True) -> bool:
    prompt = "[Y/n]" if default else "[y/N]"
    inp = input(query + " " + prompt + " ")

    return default if not inp else inp.lower()[0] == "y"


def query_save_csv_path(default: str = "output.csv"):
    path = input("Output CSV name (default is '{}'): ".format(default))

    if not path:
        path = default
    if not path.endswith(".csv"):
        path += ".csv"

    return os.path.realpath(path)


def display_results(
    valid_paths: list, results: list, display: bool = True, visualize: bool = False
):
    if not display and not bool:
        return

    for p, r in zip(valid_paths, results):
        if display:
            logger.info("\nVideo: {}\n    Result: {}".format(p, r))

        if visualize:
            v = get_entire_video_data(p)
            visualize_video_subtitle(v, r)


def write_results_to_csv(path: str, valid_paths: list, results: list):
    already_exists = os.path.exists(path)

    with open(path, "w") as f:
        writer = csv.writer(f)

        if not already_exists:
            writer.writerow(["file", "prediction"])

        for p, r in zip(valid_paths, results):
            writer.writerow([p, r])


if __name__ == "__main__":
    main()
