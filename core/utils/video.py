import os
import cv2
import numpy as np
from skvideo.io import vread
from tensorflow.keras import backend as k
from core.utils.types import Stream, Optional
from core.utils.config import LipNetConfig, DenseNetConfig


def video_to_numpy(stream: Stream, output_path: os.PathLike) -> None:
    """Save video stream into npy

    Args:
        stream (Stream): video stream, in the format of (T x H x W x C)
        output_path (os.PathLike): output target path
    """
    np.save(output_path, stream, allow_pickle=False)


def video_read(
    video_path: os.PathLike, num_frames=75, complete=False
) -> Optional[Stream]:
    """Load video to array

    Args:
        video_path (os.PathLike): video file path
        num_frames (int, optional): read first n frames. Defaults to 75.
        entire (bool): if true, ignore num_frames and load the entire video file into memory
        This may be risky when loading large video file which may leads to running out of memory

    Returns:
        Optional[Stream]: video stream, in the format of (T x W x H x C)
    """
    if complete:
        stream = vread(video_path)
    else:
        stream = vread(video_path, num_frames=num_frames)

    return video_swap_width_height(stream)


def video_read_from_npy(video_path: os.PathLike) -> Stream:
    """Load video from npy

    Args:
        video_path (os.PathLike): path to npy

    Raises:
        ValueError: Raised if target npy contains no frames

    Returns:
        Stream: Stream
    """
    stream = np.load(file=video_path, allow_pickle=False)  # T x H x W x C

    if stream.size <= 0:
        raise ValueError(f"video {video_path} is empty")

    return video_normalize(stream)


def video_normalize(stream: Stream) -> Stream:
    """Apply preprocessing on stream

    Args:
        stream (Stream): Stream

    Returns:
        Stream: Stream
    """
    return stream.astype(np.float32) / 255.0


def video_swap_width_height(stream: Stream) -> Stream:
    """Move stream axes according to keras backend configuration

    Args:
        stream (Stream): Stream

    Returns:
        Stream: Stream
    """
    return np.swapaxes(stream, 1, 2)  # T x W x H x C


def preprocess_input(stream: Stream, model: str = "lipnet") -> Stream:
    """preprocess input video sequence according to selected models

    Args:
        stream (Stream): video sequence
        model (Str, optional): model type. Defaults to "lipnet".

    Returns:
        Stream: video sequence
    """

    if model == "lipnet":
        config = LipNetConfig()
    elif model == "densenet":
        config = DenseNetConfig()
    else:
        raise TypeError("model should be either lipnet or densenet")

    frame_shape = (config.image_width, config.image_height, config.image_channels)

    return np.array(
        [cv2.resize(frame, (frame_shape[1], frame_shape[0])) for frame in stream],
        dtype=np.int32,
    )
