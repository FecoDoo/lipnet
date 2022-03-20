import numpy as np
import os
from tensorflow.keras import backend as k


def get_video_data_from_file(path: os.PathLike) -> np.ndarray:
    video_data = np.load(file=str(path), allow_pickle=False)  # T x H x W x C
    assert video_data.size != 0
    return reshape_and_normalize_video_data(video_data)


def reshape_and_normalize_video_data(video_data: np.ndarray) -> np.ndarray:
    return normalize_video_data(reshape_video_data(video_data))


def reshape_video_data(video_data: np.ndarray) -> np.ndarray:
    reshaped_video_data = np.swapaxes(video_data, 1, 2)  # T x W x H x C

    if k.image_data_format() == "channels_first":
        reshaped_video_data = np.rollaxis(reshaped_video_data, 3)  # C x T x W x H

    return reshaped_video_data


def normalize_video_data(video_data: np.ndarray) -> np.ndarray:
    return video_data.astype(np.float32) / 255
