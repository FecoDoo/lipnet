import os
import dlib
from common.files import (
    get_file_name,
    is_file,
    make_dir_if_not_exists,
)
from preprocessing.extract_roi import video_to_frames


def extract(
    group_path: os.PathLike,
    root_path: os.PathLike,
    config: dict,
    logger,
):
    groupname = group_path.name
    output_path = root_path.joinpath(config["output_path"])
    predictor_path = root_path.joinpath(config["predictor_path"])
    error_log_path = root_path.joinpath(f"logs/{groupname}.log")

    pattern = config["pattern"]

    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(str(predictor_path))

    video_target_dir = output_path.joinpath(groupname)
    make_dir_if_not_exists(video_target_dir)

    videos_failed = []

    logger.debug(f"start process {groupname}")

    for file in group_path.glob(pattern):

        video_file_name = get_file_name(file)
        video_target_path = video_target_dir.joinpath(video_file_name + ".npy")

        if is_file(video_target_path):
            logger.info(f"Video {video_target_path} already exists, skip preprocessing")
            continue

        if not video_to_frames(file, video_target_path, detector, predictor):
            videos_failed.append(video_target_path)

    with open(error_log_path, "w") as f:
        f.writelines(videos_failed)

    logger.info(f"{group_path.name} completed.")
