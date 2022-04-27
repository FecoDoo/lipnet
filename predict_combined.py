import csv

import os
import numpy as np
from skvideo.io import vread
from dotenv import load_dotenv

# load env variables
assert load_dotenv(".env")

from pathlib import Path
from common.decode import create_decoder
from core.utils.video import video_read, video_normalize
from core.utils.visualization import visualize_video_subtitle
from core.utils.types import Frame
from core.utils.config import LipNetConfig
from core.models.lipnet import LipNet
from core.models.baseline import DenseNet
from utils.converter import roi_of_all_frames
from utils.logger import get_logger


# define paths
ROOT = Path(__path__).parent.resolve()
DICTIONARY_PATH = ROOT.joinpath(os.environ["DICTIONARY_PATH"])
VIDEO_PATH = ROOT.joinpath(os.environ["VIDEO_PATH"])
LIPNET_WEIGHT_PATH = ROOT.joinpath(os.environ["LIPNET_WEIGHT_PATH"])
DENSENET_WEIGHT_PATH = ROOT.joinpath(os.environ["DENSENET_WEIGHT_PATH"])

logger = get_logger("dnn")


def main():
    """
    Entry point of the script for using a trained model for predicting videos.
    """

    logger = get_logger("DNN")

    # load model
    lipnet_weight = LIPNET_WEIGHT_PATH
    densenet_model_path = DENSENET_WEIGHT_PATH
    video_path = VIDEO_PATH

    if not lipnet_weight.is_file() or lipnet_weight.suffix != ".h5":
        logger.error("model path does not exist")
        return

    if not densenet_model_path.is_file() or densenet_model_path.suffix != ".h5":
        logger.error("model path does not exist")
        return

    if not video_path.is_file() and not video_path.is_dir():
        logger.error("video path does not exist")
        return

    # define params
    lipnet_config = LipNetConfig(
        lipnet_weight,
        video_path,
        int(os.environ["FRAME_COUNT"]),
        int(os.environ["IMAGE_WIDTH"]),
        int(os.environ["IMAGE_HEIGHT"]),
        int(os.environ["IMAGE_CHANNELS"]),
        int(os.environ["MAX_STRING"]),
    )

    predict(lipnet_config, logger)


def predict(config: LipNetConfig, logger):
    logger.info("Loading weights at: {}".format(config.model_weight_path))
    logger.info("Making predictions...")

    # init lipnet
    lipnet = LipNet(
        config.frame_count,
        config.image_channels,
        config.image_height,
        config.image_width,
        config.max_string,
    )
    lipnet.compile()
    lipnet.load_weights(config.model_weight_path)

    # init densenet
    densenet = DenseNet()
    densenet.compile()
    densenet.load_weights(config.model_weight_path)

    valid_paths = []
    input_lengths = []
    predictions = None

    elapsed_videos = 0

    video_paths = config.video_path.glob("*" + os.environ["VIDEO_SUFFIX"])

    for paths, lengths, y_pred in predict_video(lipnet, video_paths):
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


def get_video_data(path: os.PathLike) -> Frame:
    stream = video_read(path)

    cropped_stream = roi_of_all_frames(stream)

    return video_normalize(cropped_stream) if cropped_stream is not None else None


def get_entire_video_data(path: os.PathLike, num_frames=75) -> Frame:
    return np.swapaxes(vread(path, num_frames=num_frames), 1, 2)


def predict_video(lipnet: LipNet, video_paths: os.PathLike):

    input_data = [(p, get_video_data(p)) for p in video_paths]
    input_data = [x for x in input_data if x[1] is not None]

    assert len(input_data) > 0

    valid_paths = [x[0] for x in input_data]

    x_data = np.array([x[1] for x in input_data])
    lengths = [len(x) for x in x_data]

    y_pred = lipnet.predict(x_data)

    yield (valid_paths, lengths, y_pred)


def display_results(
    valid_paths: list, results: list, display: bool = True, visualize: bool = False
):
    if not display and not bool:
        return

    for p, r in zip(valid_paths, results):
        if display:
            logger.info("\nVideo: {}\nResult: {}".format(p, r))

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
