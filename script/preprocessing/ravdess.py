import sys
import os
from shutil import rmtree
from pathlib import Path
import numpy as np

root = Path(__file__).parents[2].resolve()
sys.path.insert(0, str(root))

from multiprocessing import Pool, log_to_stderr
from common.face_detection import recognition
from core.utils.video import video_read, video_to_numpy, video_transform
from core.utils.types import List, Path
from core.utils.config import emotion
from skvideo.io import ffprobe
from traceback import format_exc

data_dir = root.joinpath("data/ravdess/videos")
error_dir = root.joinpath("data/ravdess/errors")
npy_dir = root.joinpath("data/ravdess/npy")
face_npy_dir = npy_dir.joinpath("faces")
lip_npy_dir = npy_dir.joinpath("lips")

if error_dir.exists():
    rmtree(error_dir)

error_dir.mkdir(exist_ok=False)

if npy_dir.exists():
    rmtree(npy_dir)

npy_dir.mkdir(exist_ok=False)

if face_npy_dir.exists():
    rmtree(face_npy_dir)

face_npy_dir.mkdir(exist_ok=False)

if lip_npy_dir.exists():
    rmtree(lip_npy_dir)

lip_npy_dir.mkdir(exist_ok=False)

for e in emotion.keys():
    face_npy_dir.joinpath(e).mkdir(exist_ok=False)
    lip_npy_dir.joinpath(e).mkdir(exist_ok=False)

logger = log_to_stderr("INFO")


def convert(video_paths: List[Path]):
    fails = []

    logger.info(f"process {os.getpid()} started")

    for path in video_paths:
        try:
            total_frames = int(ffprobe(path).get("video").get("@nb_frames"))
            stream = video_read(path, num_frames=total_frames - 1)

            # generating one-hot encoded labels
            label = np.zeros(shape=(7,))
            label[emotion[path.parent.stem]] = 1

            faces = []
            lips = []

            # detecting faces and lips
            for frame in stream:
                detection = recognition(frame)

                face = detection[0]
                lip = detection[1]

                if face is None or lip is None:
                    raise ValueError("no face detected")

                faces.append(face)
                lips.append(lip)

            # resizing
            faces = video_transform(faces, dsize=(224, 224))
            lips = video_transform(lips, dsize=(50, 100))

            video_to_numpy(
                output_path=face_npy_dir.joinpath(path.parent.stem, path.stem + ".npy"),
                stream=faces,
            )
            video_to_numpy(
                output_path=lip_npy_dir.joinpath(path.parent.stem, path.stem + ".npy"),
                stream=lips,
            )

        except Exception:
            fails.append(str(path) + "\n")
            logger.error(format_exc())
            continue

    # record failed videos
    if fails:
        with open(error_dir.joinpath(str(os.getpid()) + ".log"), "w") as fp:
            fp.writelines(fails)


if __name__ == "__main__":

    n_proc = os.cpu_count()

    video_paths = list(data_dir.rglob("*.mp4"))

    n_video_per_proc = len(video_paths) // n_proc

    with Pool(processes=n_proc) as pool:
        try:
            handlers = [
                pool.apply_async(
                    func=convert,
                    args=(
                        video_paths[
                            idx * n_video_per_proc : (idx + 1) * n_video_per_proc
                        ],
                    ),
                )
                for idx in range(n_proc)
            ]

            for proc in handlers:
                proc.get()

            convert(video_paths=video_paths[n_proc * n_video_per_proc :])

        except KeyboardInterrupt:
            pool.terminate()
            pool.close()
