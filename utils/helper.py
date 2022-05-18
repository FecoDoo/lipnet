import os
import csv
from core.utils.visualization import visualize_video_subtitle
from core.utils.video import video_read
from core.utils.types import List, Path


def display_results(
    valid_paths: List[Path],
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

