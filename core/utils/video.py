import cv2
import warnings
import numpy as np
import random
from pathlib import Path
from functools import wraps
from skvideo.io import vread, ffprobe
from core.utils.types import Stream, Optional, Frame, Union, Tuple, List, Path


def video_input_validation(func):
    @wraps(func)
    def wrapper(stream: Stream, *args, **kwargs):
        if not isinstance(stream, np.ndarray):
            raise ValueError(f"{func.__name__}: input must be numpy array")

        return func(stream=stream, *args, **kwargs)

    return wrapper


@video_input_validation
def video_to_numpy(stream: Stream, output_path: Path) -> None:
    """Save video stream into npy

    Args:
        stream (Stream): video stream, in the format of (T x H x W x C)
        output_path (Path): output target path
    """
    np.save(output_path, stream, allow_pickle=False)


def video_read(
    video_path: Path, num_frames: int = 75, complete: bool = False
) -> Stream:
    """Load video to array

    Args:
        video_path (Path): video file path
        num_frames (int, optional): read first n frames. Defaults to 75.
        entire (bool): if true, ignore num_frames and load the entire video file into memory
        This may be risky when loading large video file which may leads to running out of memory

    Returns:
        Optional[Stream]: video stream, in the format of (T x H x W x C)
    """
    if not isinstance(video_path, Path):
        video_path = Path(video_path)

    if not video_path.is_file():
        raise ValueError(f"{video_path} does not exist")

    if complete:
        if metadata := ffprobe(video_path):
            if video_info := metadata.get("video"):
                num_frames = video_info.get("@nb_frames")

        if num_frames is None:
            raise ValueError(f"could not read metadata from {video_path}")

        num_frames = int(num_frames)

        if num_frames > 1000:
            warnings.warn(
                f"video {video_path} contains more than 1000 frames, which might lead to OOM error"
            )

        stream = vread(fname=video_path, num_frames=int(num_frames) - 1)
    else:
        stream = vread(fname=video_path, num_frames=num_frames)

    return stream


def video_read_from_npy(video_path: Path) -> Stream:
    """Load video from npy

    Args:
        video_path (Path): path to npy

    Raises:
        ValueError: Raised if target npy contains no frames

    Returns:
        Stream: Stream
    """
    stream = np.load(file=video_path, allow_pickle=False)

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
    return (stream.astype(np.float32) / 255.0).astype(np.uint8)


@video_input_validation
def video_swap_axis(stream: Stream) -> Stream:
    """Swap stream axes

    Args:
        stream (Stream): video stream

    Returns:
        Stream: video stream
    """
    res = np.swapaxes(stream, 1, 2)

    return res


@video_input_validation
def video_flip(stream: Stream):
    """flip a video sequence horizontally

    Args:
        stream (Stream): video stream

    Returns:
        Stream: video stream
    """
    res = np.flip(m=stream, axis=1)  # H x W x C

    return res


# @video_input_validation
def video_transform(
    stream: Union[Stream, List[Frame]], dsize: Tuple[int, int]
) -> Stream:
    """preprocess input video sequence according to selected models

    Args:
        stream (Stream): video sequence
        model (Str, optional): model type. Defaults to "lipnet".

    Returns:
        Stream: video sequence
    """
    if isinstance(dsize, tuple):
        # cv.resize consider frame as W x H
        dsize = (dsize[1], dsize[0])
    else:
        raise ValueError(f"dsize is not tuple, got {type(dsize)}: {dsize}")

    return np.stack([cv2.resize(src=frame, dsize=dsize) for frame in stream])


@video_input_validation
def video_sampling_frames(stream: Stream, num_frames: int = 75) -> Stream:
    """Sample new stream of size num_frames from original stream

    Args:
        stream (Stream): input stream
        num_frames (int, optional): desired output stream size. Defaults to 75.

    Returns:
        Stream: sampled stream
    """

    n = stream.shape[0]

    if n < num_frames:
        return video_sampling_frames(stream=np.repeat(stream, 2), num_frames=num_frames)
    else:
        idx = random.sample(population=range(0, n), k=num_frames)
        idx.sort()
        return stream[idx]


def video_padding_frames(stream: Stream) -> Stream:
    """Pad stream and replace None frame with the next non-empty frame

    Args:
        stream (Stream): input stream

    Returns:
        Stream: padded stream
    """

    length = len(stream)

    mask = []
    for idx, value in enumerate(stream.tolist()):
        if value is None:
            mask.append(idx)

    mask_next = find_next_non_empty_frame(mask, length - 1)
    mask_prev = find_prev_non_empty_frame(mask)

    for idx, idx_prev, idx_next in zip(mask, mask_prev, mask_next):
        # prev and next frames cannot be both None
        if idx_prev is None:
            stream[idx] = stream[idx_next]
        elif idx_next is None:
            stream[idx] = stream[idx_prev]
        else:
            # randomly fill the empty slot with prev or next frame
            choice = np.random.choice([0, 1])[0]
            stream[idx] = stream[idx_prev] if choice else stream[idx_next]

    return stream


def find_next_non_empty_frame(mask: List[int], max_idx: int) -> List[int]:
    """find next non-empty frame index regarding of current index respectively

    Args:
        mask (list): list of empty frame index
        max_idx (int): last index of the stream

    Returns:
        List[int]: list of next non-empty frame index regarding of current index
    """
    res = []

    for idx in mask:
        if idx == max_idx:
            res.append(None)
            break
        pointer = idx
        while pointer in mask:
            pointer = pointer + 1
        res.append(pointer)

    return res


def find_prev_non_empty_frame(mask: List[int]):
    """find previous non-empty frame index regarding of current index respectively

    Args:
        mask (list): list of empty frame index

    Returns:
        List[int]: list of previous non-empty frame index regarding of current index
    """
    res = []

    for idx in mask:
        if idx == 0:
            res.append(None)
            continue
        pointer = idx
        while pointer in mask:
            pointer = pointer - 1
        res.append(pointer)

    return res
