import os
import pickle
import random
from core.utils.types import List, Tuple, Path
from core.generators.baseline_generator import BatchGenerator
from utils.logger import get_logger

logger = get_logger("Dataset")


class DatasetGenerator(object):
    def __init__(
        self,
        dataset_path: Path,
        num_frames: int = 1,
        val_split: float = 0.15,
        use_cache: bool = True,
    ):
        self.dataset_path = dataset_path

        assert self.dataset_path.exists()

        self.video_dir = dataset_path.joinpath("videos")
        self.face_dir = dataset_path.joinpath("npy/faces")
        self.lip_dir = dataset_path.joinpath("npy/lips")
        self.num_frames = num_frames
        self.val_split = val_split
        self.use_cache = use_cache

        self.train = None
        self.validation = None

        self.build_dataset()

    def build_dataset(self):
        cache_path = self.dataset_path.joinpath(".cache")

        if self.use_cache and cache_path.is_file():
            logger.info(f"Loading dataset list from {cache_path}")

            with open(cache_path, "rb") as fp:
                train_faces, train_lips, val_faces, val_lips = pickle.load(fp)

        else:
            logger.info("Enumerating dataset list from disk")

            groups = self.generate_batch_from_each_group_with_shuffle(self.video_dir)

            train, val = self.split_batch(groups, self.val_split)

            train_faces, train_lips = self.locate_face_and_lip_paths(train)

            val_faces, val_lips = self.locate_face_and_lip_paths(val)

            with open(cache_path, "wb") as fp:
                pickle.dump(obj=(train_faces, train_lips, val_faces, val_lips), file=fp)

        logger.info("Found {} samples for training".format(len(train_faces)))
        logger.info("Found {} samples for validation".format(len(val_faces)))

        self.train = BatchGenerator(
            train_faces, self.num_frames
        )
        self.validation = BatchGenerator(
            val_faces, self.num_frames
        )

    @staticmethod
    def get_video_files_in_dir(path: Path) -> List[Path]:
        return list(path.glob("*.avi")) + list(path.glob("*.mp4"))

    def locate_face_and_lip_paths(self, paths: List[Path]):
        face_paths = []
        lip_paths = []

        for path in paths:
            identifier = os.path.join(path.parent.name, path.stem + ".npy")

            if self.face_dir.joinpath(identifier).exists():
                face_paths.append(self.face_dir.joinpath(identifier))
                lip_paths.append(self.lip_dir.joinpath(identifier))

        return face_paths, lip_paths

    def generate_batch_from_each_group_with_shuffle(self, path: Path) -> List[list]:
        """
        Load video file paths of each emotion group into list and return a list of lists
        """
        groups = []

        for sub_dir in [x for x in path.iterdir() if x.is_dir()]:
            numpy_file_path_list = self.get_video_files_in_dir(sub_dir)

            random.shuffle(numpy_file_path_list)

            groups.append(numpy_file_path_list)

        return groups

    @staticmethod
    def split_batch(groups: list, val_split: float) -> Tuple[list, list]:
        """Split video recordings of each group into training set and validation set.

        Args:
            groups (list): List of group names
            val_split (float): size of validation set (%)

        Returns:
            Tuple[list, list]: A tuple consists of training set and validation set (paths)
        """
        train = []
        val = []

        for group in groups:
            n_samples = len(group)
            val_amount = int(round(n_samples * val_split))

            val += group[:val_amount]
            train += group[val_amount:]

        return train, val
