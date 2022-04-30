import os
import pandas as pd
from pathlib import Path
from PIL import Image
from pandarallel import pandarallel


root = Path(__file__).parents[2].resolve()

pandarallel.initialize(progress_bar=True)

data_dir = root.joinpath("data/expw")
save_dir = data_dir.joinpath("face")
save_dir.mkdir(exist_ok=True)

# load dataset infomation
info = pd.read_csv(data_dir.joinpath("info.csv"))


def detect_face_and_save(image_name):
    file_path = data_dir.joinpath("image", image_name)
    record = info[info["image_name"] == image_name].to_numpy()

    for face in record:
        label = face[-1]

        dst = save_dir.joinpath(
            str(label) + "_" + image_name.split(".")[0] + "_" + str(face[1]) + ".jpg",
        )

        if os.path.exists(dst):
            continue

        img = Image.open(file_path).crop(box=tuple(face[2:6]))

        img.save(dst)


if __name__ == "__main__":
    _ = info["image_name"].parallel_apply(detect_face_and_save)
