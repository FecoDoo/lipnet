from multiprocessing.sharedctypes import Value
import os
import cv2
import warnings
import numpy as np
import random
from pathlib import Path
from functools import wraps
from skvideo.io import vread, ffprobe
from core.utils.types import Stream, Optional
from core.utils.config import LipNetConfig, BaselineConfig


def video_input_validation(func):
    @wraps(func)
    def wrapper(stream: Stream, *args, **kwargs):
        if not isinstance(stream, np.ndarray):
            raise ValueError(f"{func.__name__}: input must be numpy array")

        return func(stream=stream, *args, **kwargs)

    return wrapper


@video_input_validation
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
        Optional[Stream]: video stream, in the format of (T x H x W x C)
    """
    if not isinstance(video_path, os.PathLike):
        video_path = Path(video_path)

    if not video_path.is_file():
        raise ValueError(f"{video_path} does not exist")

    if complete:
        metadata = ffprobe(video_path)

        if metadata is None:
            raise ValueError(f"could not read metadata from {video_path}")

        num_frames = metadata.get("video").get("@nb_frames")

        if num_frames is None:
            raise TypeError(f"metadata corrupted for {video_path}")

        num_frames = int(num_frames)

        if num_frames > 1000:
            warnings.warn(
                f"video {video_path} contains more than 1000 frames, which might lead to OOM error"
            )

        stream = vread(fname=video_path, num_frames=int(num_frames) - 1)
    else:
        stream = vread(fname=video_path, num_frames=num_frames)

    return stream


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

    return stream


@video_input_validation
def video_normalize(stream: Stream) -> Stream:
    """Apply preprocessing on stream

    Args:
        stream (Stream): Stream

    Returns:
        Stream: Stream
    """
    return stream.astype(np.float32) / 255.0


@video_input_validation
def video_swap_width_height(stream: Stream) -> Stream:
    """Move stream axes according to keras backend configuration

    Args:
        stream (Stream): Stream

    Returns:
        Stream: Stream
    """
    return np.swapaxes(stream, 1, 2)  # T x W x H x C


@video_input_validation
def video_transform(stream: Stream, model: str = "lipnet") -> Stream:
    """preprocess input video sequence according to selected models

    Args:
        stream (Stream): video sequence
        model (Str, optional): model type. Defaults to "lipnet".

    Returns:
        Stream: video sequence
    """

    if model == "lipnet":
        config = LipNetConfig()
    elif model == "baseline":
        config = BaselineConfig()
    else:
        raise TypeError("model should be either lipnet or baseline")

    frame_shape = (config.image_height, config.image_width, config.image_channels)

    return np.array(
        [cv2.resize(frame, (frame_shape[0], frame_shape[1])) for frame in stream],
        dtype=np.uint8,
    )


@video_input_validation
def video_sample_frames(stream: Stream, num_frames: int = 75) -> Stream:
    """Sample new stream of size num_frames from original stream

    Args:
        stream (Stream): input stream
        num_frames (int, optional): desired output stream size. Defaults to 75.

    Returns:
        Stream: sampled stream
    """

    n = stream.shape[0]

    if n < num_frames:
        return video_sample_frames(stream=np.repeat(stream, 2), num_frames=num_frames)
    else:
        idx = random.sample(population=range(0, n), k=num_frames)
        idx.sort()
        return stream[idx]
