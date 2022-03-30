from datetime import timedelta
import os
import env
import cv2
import numpy as np
import skvideo.io
import traceback
from imutils import face_utils
from typing import Optional, Tuple
from dlib import get_frontal_face_detector, shape_predictor


class Converter:
    def __init__(self, logger):
        self.logger = logger
        self.frame_shape = (env.IMAGE_HEIGHT, env.IMAGE_WIDTH, env.IMAGE_CHANNELS)
        self.image_size = (env.IMAGE_HEIGHT, env.IMAGE_WIDTH)
        self.detector = get_frontal_face_detector()
        self.predictor = shape_predictor(env.DLIB_SHAPE_PREDICTOR_PATH)

        _, (self.mouth_x_idx, self.mouth_y_idx) = list(
            face_utils.FACIAL_LANDMARKS_IDXS.items()
        )[0]

    def video_to_frames(
        self, video_path: os.PathLike, output_path: os.PathLike, verbose: bool = True
    ) -> bool:
        try:
            video_name = " | ".join(str(video_path).split(".")[0].split("/")[-2:])

            if verbose:
                self.logger.info(video_name + " | processing")

            video_data = self.extract_video_data(video_path)

            np.save(output_path, video_data, allow_pickle=False)
            return True
        except ValueError as e:
            self.logger.error("{}: {}".format(video_name, str(e)))
            return False
        except Exception:
            self.logger.error(traceback.format_exc())
            return False

    def extract_video_data(self, video_path: os.PathLike) -> Optional[np.ndarray]:

        video_stream = skvideo.io.vread(str(video_path))
        video_stream_length = len(video_stream)

        if video_stream_length != env.FRAME_COUNT:
            raise ValueError("Invalid number of frames: {}".format(video_stream_length))

        mouth_data = list(map(self.extract_mouth_in_frame, enumerate(video_stream)))

        return np.array(mouth_data)

    def extract_mouth_in_frame(
        # self, frame: np.ndarray, idx: int
        self,
        var: Tuple[int, np.ndarray],
    ) -> Optional[np.ndarray]:
        m_points = self.extract_mouth_points(var[1])

        if m_points is None:
            raise ValueError("No ROI found at frame {}".format(var[0]))

        m_center = self.get_mouth_points_center(m_points)
        s_m_center = self.swap_center_axis(m_center)
        crop = self.crop_image(var[1], s_m_center, self.image_size)

        if crop.shape != self.frame_shape:
            raise ValueError("Wrong shape {} at frame {}".format(crop.shape, var[0]))

        return crop

    def extract_mouth_points(self, frame: np.ndarray) -> Optional[np.ndarray]:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        detected = self.detector(gray, 1)

        if len(detected) <= 0:
            return None

        shape = face_utils.shape_to_np(self.predictor(gray, detected[0]))

        return shape[self.mouth_x_idx : self.mouth_y_idx]

    @staticmethod
    def crop_image(image: np.ndarray, center: tuple, size: tuple) -> np.ndarray:
        start = tuple(a - b // 2 for a, b in zip(center, size))
        end = tuple(a + b for a, b in zip(start, size))
        slices = tuple(slice(a, b) for a, b in zip(start, end))

        return image[slices]

    @staticmethod
    def swap_center_axis(array: np.ndarray) -> tuple:
        return array[1], array[0]

    @staticmethod
    def get_mouth_points_center(mouth_points: np.ndarray) -> np.ndarray:
        return np.mean(mouth_points, axis=0, dtype=int)
