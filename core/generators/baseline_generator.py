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
    def __init__(
        self,
        face_paths: List[Path],
        num_frames: int = 32,
        face_frame_shape: Tuple[int, int] = (224, 224),
    ):
        super().__init__()

        self.num_frames = num_frames
        self.face_paths = face_paths

        self.face_frame_shape = face_frame_shape

        self.n_samples = len(self.face_paths)

        self.generator_steps = self.n_samples

        self.face_batch_shape = (
            num_frames,
            self.face_frame_shape[0],
            self.face_frame_shape[1],
        )

    def __len__(self) -> int:
        return self.generator_steps

    def __getitem__(self, idx: int) -> Tuple[list, list]:
        face_batch = self.face_paths[idx]
        
        # read video file and randomly choose n_frames from video sequence
        face_stream = np.load(self.face_paths[idx], allow_pickle=False)

        # record results
        faces = video_sampling_frames(
            stream=face_stream,
            num_frames=self.num_frames,
            enable_random=False
        )
        
        return faces
