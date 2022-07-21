import numpy as np
import math
import os
from tensorflow.keras.utils import Sequence
from core.utils.video import video_flip
from core.utils.types import List, Tuple, Path, Stream, Labels


class BatchGenerator(Sequence):

    __video_mean = np.array(
        [
            float(os.environ["MEAN_R"]),
            float(os.environ["MEAN_G"]),
            float(os.environ["MEAN_B"]),
        ],
        dtype=np.float32,
    )
    __video_std = np.array(
        [
            float(os.environ["STD_R"]),
            float(os.environ["STD_G"]),
            float(os.environ["STD_B"]),
        ],
        dtype=np.float32,
    )

    def __init__(self, video_paths: List[Path], align_hash: dict, batch_size: int):
        super().__init__()

        self.video_paths = video_paths
        self.align_hash = align_hash
        self.batch_size = batch_size

        self.n_videos = len(self.video_paths)
        self.n_videos_per_batch = math.ceil(self.batch_size / 2)

        self.generator_steps = math.ceil(self.n_videos / self.n_videos_per_batch)

    def __len__(self) -> int:
        return self.generator_steps

    def __getitem__(self, idx: int) -> Tuple[list, list]:
        split_start = idx * self.n_videos_per_batch
        split_end = split_start + self.n_videos_per_batch

        if split_end > self.n_videos:
            split_end = self.n_videos

        videos_batch = self.video_paths[split_start:split_end]
        videos_taken = len(videos_batch)

        videos_to_augment = self.batch_size - videos_taken

        x_data = []
        y_data = []
        input_length = []
        label_length = []
        sentences = []

        for path in videos_batch:
            stream, sentence, labels, length = self.get_data_from_path(path)

            x_data.append(stream)
            y_data.append(labels)
            label_length.append(length)
            input_length.append(len(stream))
            sentences.append(sentence)

            if videos_to_augment > 0:
                videos_to_augment -= 1

                video_argument = video_flip(stream=stream)

                x_data.append(video_argument)
                y_data.append(labels)
                label_length.append(length)
                input_length.append(len(video_argument))
                sentences.append(sentence)

        batch_size = len(x_data)
        print(batch_size)

        x_data = np.stack(arrays=x_data, axis=0)
        x_data = self.standardize_batch(x_data)

        y_data = np.array(y_data, np.uint8)
        input_length = np.array(input_length, dtype=np.uint8)
        label_length = np.array(label_length, dtype=np.uint8)
        sentences = np.array(sentences)

        inputs = [x_data, y_data, input_length, label_length]

        # dummy data for dummy loss function
        outputs = np.zeros(shape=(batch_size,), dtype=np.uint8)

        return inputs, outputs

    def get_data_from_path(self, path: Path) -> Tuple[Stream, str, Labels, int]:
        align = self.align_hash[path.stem]
        return (
            np.load(path, allow_pickle=False),
            align.sentence,
            align.labels,
            align.length,
        )

    def standardize_batch(self, batch: List[Stream]) -> List[Stream]:
        return (batch - self.__video_mean) / (self.__video_std + 1e-6)
