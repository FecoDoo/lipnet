import numpy as np
import os
from tensorflow.keras.utils import Sequence
from tensorflow import dtypes, cast, one_hot
from common.face_detection import recognition
from core.utils.video import video_read, video_sample_frames, video_normalize
from core.utils.types import List, Tuple
from core.utils.config import emotion
from core.utils.types import Stream


class VideoSampleGenerator(Sequence):
    def __init__(self, video_path: os.PathLike, batch_size: int, num_frames: int = 75):
        super().__init__()
        self.num_frames = num_frames
        self.video_path = video_path
        self.batch_size = batch_size

    def __len__(self) -> int:
        return self.batch_size

    def __getitem__(self) -> Tuple[list, list]:
        faces = []
        lips = []
        labels = []

        stream = video_read(video_path=self.video_path, complete=True)
        label = one_hot(
                indices=cast(emotion[self.video_path.parent.name], dtype=dtypes.uint8), depth=7
            )
        # shuffle one video stream into batch_size samples
        for n_batch in range(self.batch_size):
            sampled_stream = video_sample_frames(
                stream=stream, num_frames=self.num_frames
            )
            
            res = [recognition(frame) for frame in sampled_stream]
            
            faces.append([i[0] for i in res])
            lips.append([i[1] for i in res])
            labels.append(label)

        inputs = [faces, lips]

        return inputs, labels
