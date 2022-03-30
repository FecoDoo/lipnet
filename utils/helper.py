import os
import csv
import numpy as np
import skvideo.io
from typing import List
from core.helpers.video import (
    get_video_data_from_file,
    reshape_and_normalize_video_data,
)
from core.utils.visualization import visualize_video_subtitle


def display_results(
    valid_paths: list, results: list, display: bool = True, visualize: bool = False
):
    if not display and not bool:
        return

    for p, r in zip(valid_paths, results):
        if display:
            print("\nVideo: {}\n    Result: {}".format(p, r))

        if visualize:
            v = get_entire_video_data(p)
            visualize_video_subtitle(v, r)


def query_save_csv_path(default: str = "output.csv"):
    path = input("Output CSV name (default is '{}'): ".format(default))

    if not path:
        path = default
    if not path.endswith(".csv"):
        path += ".csv"

    return os.path.realpath(path)


def query_yes_no(query: str, default: bool = True) -> bool:
    prompt = "[Y/n]" if default else "[y/N]"
    inp = input(query + " " + prompt + " ")

    return default if not inp else inp.lower()[0] == "y"


def write_results_to_csv(path: os.PathLike, valid_paths: list, results: list):
    already_exists = path.exists()

    with open(path, "w") as f:
        writer = csv.writer(f)

        if not already_exists:
            writer.writerow(["file", "prediction"])

        for p, r in zip(valid_paths, results):
            writer.writerow([p, r])


def get_entire_video_data(path: os.PathLike) -> np.ndarray:
    if path.suffix == ".mpg":
        return np.swapaxes(skvideo.io.vread(path), 1, 2)
    else:
        return get_video_data_from_file(path)
