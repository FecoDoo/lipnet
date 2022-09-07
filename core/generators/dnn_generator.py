import numpy as np
import math
import os
from tensorflow.keras.utils import Sequence
from core.utils.video import video_sampling_frames
from core.utils.types import List, Tuple, Stream, Path
from core.utils.config import emotion
from utils.logger import get_logger

logger = get_logger("Generator")


class BatchGenerator(Sequence):

    __video_mean = np.array(
        [
            float(os.environ["MEAN_R"]),
            float(os.environ["MEAN_G"]),
            float(os.environ["MEAN_B"]),
        ]
    )
    __video_std = (
        np.array(
            [
                float(os.environ["STD_R"]),
                float(os.environ["STD_G"]),
                float(os.environ["STD_B"]),
            ]
        )
        + 1e-6
    )

    def __init__(
        self,
        face_paths: List[Path],
        lip_paths: List[Path],
        batch_size: int = 4,
        num_frames: int = 75,
        face_frame_shape: Tuple[int, int] = (224, 224),
        lip_frame_shape: Tuple[int, int] = (50, 100),
    ):
        super().__init__()

        assert len(face_paths) == len(lip_paths)

        self.num_frames = num_frames
        self.face_paths = face_paths
        self.lip_paths = lip_paths
        self.batch_size = batch_size

        self.face_frame_shape = face_frame_shape
        self.lip_frame_shape = lip_frame_shape

        self.n_samples = len(self.face_paths)

        self.generator_steps = math.ceil(self.n_samples / self.batch_size)

        self.face_batch_shape = (
            num_frames,
            self.face_frame_shape[0],
            self.face_frame_shape[1],
        )

    def __len__(self) -> int:
        return self.generator_steps

    def __getitem__(self, idx: int) -> Tuple[list, list]:
        # get split slice
        split_start = idx * self.batch_size
        split_end = split_start + self.batch_size

        if split_end > self.n_samples:
            split_end = self.n_samples

        face_batch = self.face_paths[split_start:split_end]
        lip_batch = self.lip_paths[split_start:split_end]

        # define containers
        faces = []
        lips = []
        labels = []

        # shuffle one video stream into batch_size samples
        for face_path, lip_path in zip(face_batch, lip_batch):
            # read video file and randomly choose n_frames from video sequence
            face_stream = np.load(face_path, allow_pickle=False)
            lip_stream = np.load(lip_path, allow_pickle=False)

            # read label
            label_char = face_path.parent.name
            label_num = np.zeros(shape=(7,))
            label_num[emotion[label_char]] = 1

            # record results
            faces.append(
                video_sampling_frames(stream=face_stream, num_frames=self.num_frames, enable_random=False)
            )
            lips.append(
                video_sampling_frames(stream=lip_stream, num_frames=self.num_frames, enable_random=False)
            )
            labels.append(label_num)

        inputs = [np.stack(faces, axis=0), np.stack(lips, axis=0)]

        return inputs, np.stack(labels, axis=0)

    def preprocessing_lip_stream(self, batch: Stream) -> Stream:
        return (batch - self.__video_mean) / self.__video_std
