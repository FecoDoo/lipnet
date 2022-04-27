import os
import csv
from core.utils.visualization import visualize_video_subtitle
from core.utils.video import video_read
from typing import List


def display_results(
    valid_paths: List[os.PathLike],
    results: list,
    display: bool = True,
    visualize: bool = False,
):
    if not display and not bool:
        return

    for video_path, r in zip(valid_paths, results):
        if display:
            print("Video: {}\nResult: {}".format(video_path, r))

        if visualize:
            stream = video_read(video_path, complete=True)
            visualize_video_subtitle(stream, r)


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
