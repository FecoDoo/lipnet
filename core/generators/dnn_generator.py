import numpy as np
import os
from tensorflow.keras.utils import Sequence
from common.face_detection import recognition
from core.utils.video import (
    video_read,
    video_sampling_frames,
    video_padding_frames,
    video_transform,
)
from core.utils.types import List, Tuple, Stream, Frame
from core.utils.config import emotion


class VideoSampleGenerator(Sequence):

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
        video_paths: List[os.PathLike],
        batch_size: int = 4,
        num_frames: int = 75,
        face_frame_shape: Tuple[int] = (224, 224),
        lip_frame_shape: Tuple[int] = (50, 100),
    ):
        super().__init__()
        self.num_frames = num_frames
        self.video_paths = video_paths
        self.batch_size = batch_size

        self.face_frame_shape = face_frame_shape
        self.lip_frame_shape = lip_frame_shape

        self.n_videos = len(self.video_paths)
        self.n_videos_per_batch = int(np.ceil(self.batch_size / 2))

        self.generator_steps = int(np.ceil(self.n_videos / self.n_videos_per_batch))

        self.face_batch_shape = (
            num_frames,
            self.face_frame_shape[0],
            self.face_frame_shape[1],
        )
        self.face_salt = np.stack(
            [
                np.zeros(shape=self.face_batch_shape, dtype=np.float32) - 103.939,
                np.zeros(shape=self.face_batch_shape, dtype=np.float32) - 116.779,
                np.zeros(shape=self.face_batch_shape, dtype=np.float32) - 123.68,
            ],
            axis=-1,
        )

    def __len__(self) -> int:
        return self.generator_steps

    def __getitem__(self, idx: int) -> Tuple[list, list]:
        split_start = idx * self.n_videos_per_batch
        split_end = split_start + self.n_videos_per_batch

        if split_end > self.n_videos:
            split_end = self.n_videos

        videos_batch = self.video_paths[split_start:split_end]

        faces = []
        lips = []
        labels = []

        # shuffle one video stream into batch_size samples
        for path in videos_batch:
            # read video file
            stream = video_read(video_path=path, complete=True)
            # randomly choose n_frames from video sequence
            stream = video_sampling_frames(stream=stream, num_frames=self.num_frames)

            # read label
            label_char = path.parent.name
            label_num = np.zeros(shape=(7,))
            label_num[emotion[label_char]] = 1

            # face and lip detection
            res = [recognition(frame) for frame in stream]

            # padding, resizing and scaling
            face_stream = self.preprocessing_face_stream(
                video_transform(
                    stream=video_padding_frames([i[0] for i in res]),
                    dsize=self.face_frame_shape,
                )
            )

            lip_stream = self.preprocessing_lip_stream(
                video_transform(
                    stream=video_padding_frames([i[1] for i in res]),
                    dsize=self.lip_frame_shape,
                )
            )

            # record results
            faces.append(face_stream)
            lips.append(lip_stream)
            labels.append(label_num)

        inputs = [np.array(faces, dtype=np.uint8), np.array(lips, dtype=np.uint8)]

        return inputs, np.array(labels, dtype=np.uint8)

    def preprocessing_face_stream(self, batch: Stream) -> Stream:
        return batch + self.face_salt

    def preprocessing_lip_stream(self, batch: Stream) -> Stream:
        return (batch - self.__video_mean) / self.__video_std
