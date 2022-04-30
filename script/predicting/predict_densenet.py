import os
from dotenv import load_dotenv
from pathlib import Path
from core.models.baseline import DenseNet
from tensorflow import io, image, dtypes
from tensorflow import data
from utils.logger import get_logger

assert load_dotenv(".env")

# define paths
ROOT = Path(__file__).joinpath("../").resolve()
DENSENET_WEIGHT_PATH = ROOT.joinpath(os.environ["DENSENET_WEIGHT_PATH"])
IMAGE_PATH = ROOT.joinpath("data/expw/face/")

# define logger
logger = get_logger("baseline")


def preprocessing_images(path):
    img = io.read_file(path)
    img = io.decode_jpeg(img)
    img = image.convert_image_dtype(img, dtypes.float32)

    img = image.resize(img, [224, 224])
    return img


def generate_dataset(path: os.PathLike, pattern="*.jpg"):
    logger.info("Generating Dataset")

    if not path.exists():
        raise FileNotFoundError("dataset path not exists")

    dataset = data.Dataset.list_files(str(path.joinpath(pattern)))
    dataset = dataset.map(preprocessing_images)

    return dataset


def main():
    """
    Entry point of the script for using a trained model for predicting videos.
    """
    model = DenseNet()

    model.load_weights(DENSENET_WEIGHT_PATH)

    dataset = generate_dataset(IMAGE_PATH)

    sample = dataset.take(1)

    y_pred = model.predict(sample)

    # print(label)
    print(y_pred)


if __name__ == "__main__":
    main()
