import os
from tokenize import group
import env
import cv2
import numpy as np
import skvideo.io
import dlib
from colorama import Fore
from imutils import face_utils
from typing import Optional, Dict
from tqdm import tqdm

FRAME_SHAPE = (env.IMAGE_HEIGHT, env.IMAGE_WIDTH, env.IMAGE_CHANNELS)
IMAGE_SIZE = (env.IMAGE_HEIGHT, env.IMAGE_WIDTH)

INFO_LOG = Fore.BLUE
ERROR_LOG = Fore.RED
DEBUG_LOG = Fore.YELLOW
CRITICAL_LOG = Fore.GREEN

detector_type = "lvm"
dlib.DLIB_USE_CUDA = False


class ExtractorROI:
    def __init__(
        self,
        detector,
        predictor,
        logger,
    ):
        self.detector = detector
        self.predictor = predictor
        self.logger = logger

    def video_to_frames(
        self, video_path: os.PathLike, output_path: os.PathLike
    ) -> bool:
        video_data = self.extract_video_data(video_path)

        if video_data is None:
            return False
        else:
            np.save(output_path, video_data)
            return True

    def extract_video_data(
        self, video_path: os.PathLike, verbose: bool = True
    ) -> Optional[np.ndarray]:
        if verbose:
            self.logger.debug("{}\n{}".format(DEBUG_LOG, video_path.name))

        video_data = skvideo.io.vread(video_path.as_posix())
        video_data_len = len(video_data)

        if video_data_len != env.FRAME_COUNT:
            self.logger.error(
                ERROR_LOG + "Wrong number of frames: {}".format(video_data_len)
            )
            return None

        mouth_data = []

        for i, f in enumerate(video_data):
            c = self.extract_mouth_on_frame(f, i)
            if c is None:
                return None
            mouth_data.append(c)

        mouth_data = np.array(mouth_data)
        # if verbose and bar:
        #     bar.finish()

        return mouth_data

    def extract_mouth_on_frame(
        self, frame: np.ndarray, idx: int
    ) -> Optional[np.ndarray]:
        m_points = self.extract_mouth_points(frame)
        if m_points is None:
            self.logger.error(ERROR_LOG + "No ROI found at frame {}".format(idx))
            return None

        m_center = self.get_mouth_points_center(m_points)
        s_m_center = self.swap_center_axis(m_center)
        crop = self.crop_image(frame, s_m_center, IMAGE_SIZE)
        if crop.shape != FRAME_SHAPE:
            self.logger.error(
                ERROR_LOG + "Wrong shape {} at frame {}".format(crop.shape, idx)
            )
            return None

        return crop

    @staticmethod
    def crop_image(image: np.ndarray, center: tuple, size: tuple) -> np.ndarray:
        start = tuple(a - b // 2 for a, b in zip(center, size))
        end = tuple(a + b for a, b in zip(start, size))
        slices = tuple(slice(a, b) for a, b in zip(start, end))

        return image[slices]

    @staticmethod
    def swap_center_axis(t: np.ndarray) -> tuple:
        return t[1], t[0]

    @staticmethod
    def get_mouth_points_center(mouth_points: np.ndarray) -> np.ndarray:
        mouth_centroid = np.mean(mouth_points[:, -2:], axis=0, dtype=int)
        return mouth_centroid

    def extract_mouth_points(self, frame: np.ndarray) -> Optional[np.ndarray]:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        detected = self.detector(gray, 1)

        if len(detected) <= 0:
            return None

        if detector_type == "cnn":
            shape = face_utils.shape_to_np(self.predictor(gray, detected[0].rect))

        else:
            shape = face_utils.shape_to_np(self.predictor(gray, detected[0]))

        _, (i, j) = list(face_utils.FACIAL_LANDMARKS_IDXS.items())[0]

        return np.array([shape[i:j]][0])


def extract(
    group_path: os.PathLike,
    root_path: os.PathLike,
    config: Dict,
    logger,
):
    try:
        groupname = group_path.name
        output_path = root_path.joinpath(config["output_path"]).resolve()
        predictor_path = root_path.joinpath(config["predictor_path"]).resolve()
        cnn_predictor_path = root_path.joinpath(config["cnn_predictor_path"]).resolve()
        error_log_path = root_path.joinpath(
            os.path.join(config["log_path"], f"{groupname}.log")
        ).resolve()

        pattern = config["pattern"]

        if detector_type == "cnn":
            detector = dlib.cnn_face_detection_model_v1(cnn_predictor_path.as_posix())
            logger.critical(f"{CRITICAL_LOG}load cnn detector")
        else:
            detector = dlib.get_frontal_face_detector()
            logger.critical(f"{CRITICAL_LOG}load svm detector")

        predictor = dlib.shape_predictor(predictor_path.as_posix())

        video_target_dir = output_path.joinpath(groupname)

        video_target_dir.mkdir(exist_ok=True)

        videos_failed = []

        logger.info(f"{INFO_LOG}start process {groupname}")

        extractor = ExtractorROI(detector=detector, predictor=predictor, logger=logger)

        for file_path in tqdm(group_path.glob(pattern)):
            video_file_name = file_path.stem
            video_target_path = video_target_dir.joinpath(video_file_name + ".npy")

            if video_target_path.is_file():
                logger.info(
                    f"{INFO_LOG}Video {groupname + '-' + video_file_name} already exists, skip preprocessing"
                )
                continue

            if not extractor.video_to_frames(file_path, video_target_path):
                videos_failed.append(video_file_name)

        with open(error_log_path.as_posix(), "w") as f:
            f.writelines(videos_failed)

    except Exception as e:
        logger.error(f"{ERROR_LOG}{e}")
    finally:
        logger.info(f"{INFO_LOG}{group_path.name} completed.")


def extract_single(
    video_name: str,
    root_path: os.PathLike,
    group_name: os.PathLike,
    config: Dict,
    logger,
):
    try:
        output_path = root_path.joinpath(
            os.path.join(config["output_path"], group_name, video_name + ".npy")
        ).resolve()
        predictor_path = root_path.joinpath(config["predictor_path"]).resolve()
        cnn_predictor_path = root_path.joinpath(config["cnn_predictor_path"]).resolve()

        video_path = root_path.joinpath(
            os.path.join(config["dataset_path"], group_name, video_name + ".mpg")
        ).resolve()

        if detector_type == "cnn":
            detector = dlib.cnn_face_detection_model_v1(cnn_predictor_path.as_posix())
            logger.critical(f"{CRITICAL_LOG}load cnn detector")
        else:
            detector = dlib.get_frontal_face_detector()
            logger.critical(f"{CRITICAL_LOG}load svm detector")

        predictor = dlib.shape_predictor(predictor_path.as_posix())

        logger.info(f"{INFO_LOG}start process {group_name}")

        extractor = ExtractorROI(detector=detector, predictor=predictor, logger=logger)

        if output_path.is_file():
            logger.info(
                f"{INFO_LOG}Video {group_name + '-' + video_name} already exists, skip preprocessing"
            )

        if not extractor.video_to_frames(video_path, output_path):
            logger.error(f"{ERROR_LOG}video extracting error")

    except Exception as e:
        logger.error(f"{ERROR_LOG}{e}")
