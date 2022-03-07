import os
from typing import List


def is_dir(path: os.PathLike) -> bool:
    return isinstance(path, os.PathLike) and path.exists() and path.is_dir()


def is_file(path: os.PathLike) -> bool:
    return isinstance(path, os.PathLike) and path.exists() and path.is_file()


def get_file_extension(path: os.PathLike) -> os.PathLike:
    return path.suffix if is_file(path) else ""


def get_file_name(path: os.PathLike) -> os.PathLike:
    return path.stem if is_file(path) else ""


def make_dir_if_not_exists(path: os.PathLike):
    if not is_dir(path):
        os.makedirs(path)


def get_immediate_subdirs(path: os.PathLike) -> List[os.PathLike]:
    return [path.joinpath(s) for s in next(os.walk(path))[1]] if is_dir(path) else []
