import os
import pickle
import random
from core.utils.types import List, Tuple, Dict, Path
from core.generators.batch_generator import BatchGenerator
from core.utils.align import Align, align_from_file


class DatasetGenerator(object):
    def __init__(
        self,
        dataset_path: Path,
        aligns_path: Path,
        batch_size: int,
        max_string: int,
        val_split: float,
        use_cache: bool = True,
    ):
        self.dataset_path = dataset_path
        self.aligns_path = aligns_path
        self.batch_size = batch_size
        self.max_string = max_string
        self.val_split = val_split
        self.use_cache = use_cache

        self.train_generator = None
        self.val_generator = None

        self.build_dataset()

    def build_dataset(self):
        cache_path = self.dataset_path.joinpath(".cache")

        if self.use_cache and cache_path.is_file():
            print("Loading dataset list from cache...")

            with open(cache_path, "rb") as f:
                train_videos, train_aligns, val_videos, val_aligns = pickle.load(f)
        else:
            print("Enumerating dataset list from disk...")

            groups = self.generate_video_list_by_groups_with_shuffle(self.dataset_path)
            train_videos, val_videos = self.split_speaker_groups(groups, self.val_split)

            train_aligns = self.generate_align_hash(train_videos)
            val_aligns = self.generate_align_hash(val_videos)

            with open(cache_path, "wb") as f:
                pickle.dump(
                    obj=(train_videos, train_aligns, val_videos, val_aligns), file=f
                )

        print(
            "Found {} videos and {} aligns for training".format(
                len(train_videos), len(train_aligns)
            )
        )
        print(
            "Found {} videos and {} aligns for validation\n".format(
                len(val_videos), len(val_aligns)
            )
        )

        self.train_generator = BatchGenerator(
            train_videos, train_aligns, self.batch_size
        )
        self.val_generator = BatchGenerator(val_videos, val_aligns, self.batch_size)

    @staticmethod
    def get_numpy_files_in_dir(path: Path) -> List[Path]:
        return list(path.glob("*.npy"))

    def generate_video_list_by_groups_with_shuffle(self, path: Path) -> List[list]:
        """
        Load video file paths of each speaker group into list and return a list of lists
        """
        speaker_groups = []

        for sub_dir in [x for x in path.iterdir() if x.is_dir()]:
            videos_in_group = self.get_numpy_files_in_dir(sub_dir)
            random.shuffle(videos_in_group)

            speaker_groups.append(videos_in_group)

        return speaker_groups

    @staticmethod
    def split_speaker_groups(groups: list, val_split: float) -> Tuple[list, list]:
        """Split video recordings of each group into training set and validation set.

        Args:
            groups (list): List of group names
            val_split (float): size of validation set (%)

        Returns:
            Tuple[list, list]: A tuple consists of training set and validation set (paths)
        """
        train_list = []
        val_list = []

        for group in groups:
            group_len = len(group)
            val_amount = int(round(group_len * val_split))

            train_list += group[val_amount:]
            val_list += group[:val_amount]

        return train_list, val_list

    def generate_align_hash(self, videos: list) -> Dict[str, Align]:
        align_hash = {}

        for path in videos:
            video_name = path.stem
            align_path = self.aligns_path.joinpath(video_name + ".align")

            align_hash[video_name] = align_from_file(align_path, self.max_string)

        return align_hash
