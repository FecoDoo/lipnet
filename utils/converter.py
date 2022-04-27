import cv2
import os
import numpy as np
from imutils import face_utils
from dlib import get_frontal_face_detector, shape_predictor
from core.utils.types import Stream, Optional, Frame

frame_shape = (
    int(os.environ["IMAGE_WIDTH"]),
    int(os.environ["IMAGE_HEIGHT"]),
    int(os.environ["IMAGE_CHANNELS"]),
)

image_size = (int(os.environ["IMAGE_WIDTH"]), int(os.environ["IMAGE_HEIGHT"]))
detector = get_frontal_face_detector()
predictor = shape_predictor(os.environ["DLIB_SHAPE_PREDICTOR_PATH"])

landmark_name, (mouth_x_idx, mouth_y_idx) = list(
    face_utils.FACIAL_LANDMARKS_IDXS.items()
)[0]


# passed if no ROI is found or the shape after cropping is wrong
dummy_cropped_image = np.zeros(
    shape=frame_shape,
    dtype=np.float32,
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


def roi_of_each_frame(idx: int, frame: Frame) -> Optional[Frame]:
    """Crop each frame by extract ROI around subject's mouth

    Args:
        video_path (os.PathLike): path to the video file

    Returns:
        Optional[np.ndarray]: cropped frames, return None if
    """
    m_points = extract_mouth_points(frame)

    if m_points is None:
        print("No ROI found at frame {}".format(idx))
        return dummy_cropped_image

    m_center = get_mouth_points_center(m_points)

    crop = crop_image(frame, m_center, image_size)  # T x W x H x C

    if crop.shape != frame_shape:
        print("Wrong shape {} at frame {}".format(crop.shape, idx))
        return dummy_cropped_image

    return crop


def extract_mouth_points(stream: Stream) -> Optional[Stream]:
    """Extract mouth landmarks

    Args:
        stream (Stream): Stream

    Returns:
        Optional[Stream]: Stream or None
    """
    gray = cv2.cvtColor(np.swapaxes(stream, 0, 1), cv2.COLOR_BGR2GRAY)
    detected = detector(gray, 1)

    if len(detected) <= 0:
        return None

    shape = face_utils.shape_to_np(predictor(gray, detected[0]))
    return shape[mouth_x_idx:mouth_y_idx]


def crop_image(frame: Frame, center: tuple, size: tuple) -> Frame:
    """Crop frame accroding to roi

    Args:
        frame (np.ndarray): one frame
        center (tuple): center of roi
        size (tuple): size of roi

    Returns:
        Frame: roi frame after cropping
    """
    start = tuple(a - b // 2 for a, b in zip(center, size))
    end = tuple(a + b for a, b in zip(start, size))
    slices = tuple(slice(a, b) for a, b in zip(start, end))

    return frame[slices]


def get_mouth_points_center(mouth_points: np.ndarray) -> np.ndarray:
    return np.mean(mouth_points, axis=0, dtype=np.int32)
