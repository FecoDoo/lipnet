import numpy as np
import os
from tensorflow.keras.utils import Sequence
from core.utils.video import video_read
from core.utils.types import List, Tuple, Path, Stream, Labels


class BatchGenerator(Sequence):

    __video_mean = np.array(
        [os.environ.get("MEAN_R"), os.environ.get("MEAN_G"), os.environ.get("MEAN_B")]
    )
    __video_std = np.array(
        [os.environ.get("STD_R"), os.environ.get("STD_G"), os.environ.get("STD_B")]
    )

    def __init__(self, video_paths: List[Path], align_hash: dict, batch_size: int):
        super().__init__()

        self.video_paths = video_paths
        self.align_hash = align_hash
        self.batch_size = batch_size

        self.n_videos = len(self.video_paths)
        self.n_videos_per_batch = int(np.ceil(self.batch_size / 2))

        self.generator_steps = int(np.ceil(self.n_videos / self.n_videos_per_batch))

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
            video_data, sentence, labels, length = self.get_data_from_path(path)

            x_data.append(video_data)
            y_data.append(labels)
            label_length.append(length)
            input_length.append(len(video_data))
            sentences.append(sentence)

            if videos_to_augment > 0:
                videos_to_augment -= 1

                f_video_data = self.flip_video(video_data)

                x_data.append(f_video_data)
                y_data.append(labels)
                label_length.append(length)
                input_length.append(len(video_data))
                sentences.append(sentence)

        batch_size = len(x_data)

        x_data = np.array(x_data)
        x_data = self.standardize_batch(x_data)

        y_data = np.array(y_data)
        input_length = np.array(input_length)
        label_length = np.array(label_length)
        sentences = np.array(sentences)

        # inputs = {
        #     "input": x_data,
        #     "labels": y_data,
        #     "input_length": input_length,
        #     "label_length": label_length,
        #     "sentences": sentences,
        # }

        # # dummy data for dummy loss function
        # outputs = {"ctc": np.zeros([batch_size])}
        inputs = [x_data, y_data, input_length, label_length]

        # dummy data for dummy loss function
        outputs = np.zeros((batch_size))

        return inputs, outputs

    def get_data_from_path(self, path: Path) -> Tuple[Stream, str, Labels, int]:
        align = self.align_hash[path.stem]
        return (
            video_read(path),
            align.sentence,
            align.labels,
            align.length,
        )

    @staticmethod
    def flip_video(video_data: Stream) -> Stream:
        return np.flip(
            video_data, axis=1
        )  # flip in the vertical axis because videos are flipped 90deg when passed to the model

    def standardize_batch(self, batch: Stream) -> Stream:
        return (batch - self.__video_mean) / (self.__video_std + 1e-6)
