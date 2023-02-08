import sys
import os
from shutil import rmtree
from pathlib import Path
import numpy as np

root = Path(__file__).parents[2].resolve()
sys.path.insert(0, str(root))

from multiprocessing import Pool, log_to_stderr
from common.facemesh import recognition
from core.utils.video import video_read, video_to_numpy, video_transform
from core.utils.types import Path
from core.utils.config import emotion
from skvideo.io import ffprobe

data_dir = root.joinpath("data/casia/normal")
error_dir = data_dir.joinpath("errors")
npy_dir = data_dir.joinpath("npy")
face_npy_dir = npy_dir.joinpath("faces")
lip_npy_dir = npy_dir.joinpath("lips")

if error_dir.exists():
    rmtree(error_dir)

error_dir.mkdir(exist_ok=True)

if npy_dir.exists():
    rmtree(npy_dir)

npy_dir.mkdir(exist_ok=True)

if face_npy_dir.exists():
    rmtree(face_npy_dir)

face_npy_dir.mkdir(exist_ok=True)

if lip_npy_dir.exists():
    rmtree(lip_npy_dir)

lip_npy_dir.mkdir(exist_ok=True)

for e in emotion.keys():
    face_npy_dir.joinpath(e).mkdir(exist_ok=True)
    lip_npy_dir.joinpath(e).mkdir(exist_ok=True)

logger = log_to_stderr("INFO")


def convert(video_path: Path):
    try:
        # get total number of frames of each video
        total_frames = int(ffprobe(video_path).get("video").get("@nb_frames"))
        # load video
        stream = video_read(video_path, num_frames=total_frames - 1)

        # generating one-hot encoded labels
        label = np.zeros(shape=(7,))
        label[emotion[video_path.parent.stem]] = 1

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
            output_path=face_npy_dir.joinpath(
                video_path.parent.stem, video_path.stem + ".npy"
            ),
            stream=faces,
        )
        video_to_numpy(
            output_path=lip_npy_dir.joinpath(
                video_path.parent.stem, video_path.stem + ".npy"
            ),
            stream=lips,
        )

    except Exception:
        return video_path

if __name__ == "__main__":
    video_list = list(data_dir.rglob("*.avi"))
    
    with Pool(processes=os.cpu_count()) as pool:
        try:
            res = pool.map(func=convert, iterable=video_list)
            print(res)
        except KeyboardInterrupt:
            pool.terminate()
            pool.close()
