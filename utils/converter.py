import logging
import cv2
import numpy as np
from pathlib import Path
from imutils import face_utils
from dlib import get_frontal_face_detector, shape_predictor
from core.utils.types import Stream, Optional, Frame
import env

ROOT = Path(__file__).parents[1].resolve()
logger = logging.getLogger("converter")
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler())


frame_shape = (env.IMAGE_HEIGHT, env.IMAGE_WIDTH, env.IMAGE_CHANNELS)
image_size = (env.IMAGE_HEIGHT, env.IMAGE_WIDTH)
detector = get_frontal_face_detector()
predictor = shape_predictor(str(ROOT.joinpath(env.DLIB_SHAPE_PREDICTOR_PATH)))

landmark_name, (mouth_x_idx, mouth_y_idx) = list(
    face_utils.FACIAL_LANDMARKS_IDXS.items()
)[0]


# passed if no ROI is found or the shape after cropping is wrong
dummy_cropped_image = np.zeros(
    shape=(env.IMAGE_HEIGHT, env.IMAGE_WIDTH, 3), dtype=np.uint8
)


def roi_of_all_frames(stream: Stream) -> Stream:
    """Extract all roi from stream

    Args:
        stream (Stream): Stream

    Returns:
        Stream: Stream
    """
    res = []

    for idx, frame in enumerate(stream):
        res.append(roi_of_each_frame(idx, frame))

    return np.stack(res)


def roi_of_each_frame(idx: int, frame: Frame) -> Optional[Stream]:
    """Crop each frame by extract ROI around subject's mouth

    Args:
        video_path (os.PathLike): path to the video file

    Returns:
        Optional[np.ndarray]: cropped frames, return None if
    """
    m_points = extract_mouth_points(frame)

    if m_points is None:
        logger.error("No ROI found at frame {}".format(idx))
        return dummy_cropped_image

    m_center = get_mouth_points_center(m_points)  # W x H

    m_center = swap_center_axis(m_center)  # H x W

    crop = crop_image(frame, m_center, image_size)  # T x H x W x C

    if crop.shape != frame_shape:
        logger.error("Wrong shape {} at frame {}".format(crop.shape, idx))
        return dummy_cropped_image

    return crop


def extract_mouth_points(stream: Stream) -> Optional[Stream]:
    """Extract mouth landmarks

    Args:
        stream (Stream): Stream

    Returns:
        Optional[Stream]: Stream or None
    """
    gray = cv2.cvtColor(stream, cv2.COLOR_BGR2GRAY)
    detected = detector(gray, 1)

    if len(detected) <= 0:
        return None

    shape = face_utils.shape_to_np(predictor(gray, detected[0]))

    return shape[mouth_x_idx:mouth_y_idx]


def crop_image(image: Frame, center: tuple, size: tuple) -> Frame:
    """Crop frame accroding to roi

    Args:
        image (np.ndarray): one frame
        center (tuple): center of roi
        size (tuple): size of roi

    Returns:
        Frame: roi frame after cropping
    """
    start = tuple(a - b // 2 for a, b in zip(center, size))
    end = tuple(a + b for a, b in zip(start, size))
    slices = tuple(slice(a, b) for a, b in zip(start, end))

    return image[slices]


def swap_center_axis(center: np.ndarray) -> tuple:
    """swap center axis, from W x H to H x W

    Args:
        center (np.ndarray): [W, H]

    Returns:
        tuple: (H, W)
    """
    return center[1], center[0]


def get_mouth_points_center(mouth_points: np.ndarray) -> np.ndarray:
    return np.mean(mouth_points, axis=0, dtype=np.int16)
