import os
import time
from typing import NamedTuple
from datetime import datetime, timedelta
from pathlib import Path
from utils.logger import get_logger
from common.decode import create_decoder
from core.callbacks.callbacks import ErrorRates, CSVLogger, ModelCheckpoint, TensorBoard
from core.generators.dataset_generator import DatasetGenerator
from core.models.lipnet import LipNet
from dotenv import load_dotenv

assert load_dotenv(".env")

ROOT = Path(__file__).parent.resolve()

# define dataset paths
DATA_PATH = ROOT.joinpath(os.environ["GRID_NPY_PATH"])
ALIGN_PATH = ROOT.joinpath(os.environ["GRID_ALIGN_PATH"])

# define training params
LIPNET_MODEL_SAVE_PATH = ROOT.joinpath("models")
LOG_PATH = ROOT.joinpath("logs/training")

DICTIONARY_PATH = ROOT.joinpath("data/dictionaries/grid.txt")

LIPNET_MODEL_SAVE_PATH.mkdir(exist_ok=True)
LOG_PATH.mkdir(exist_ok=True)

DATETIME_FMT = os.environ["DATETIME_FMT"]


class TrainingConfig(NamedTuple):
    """Baisc Training Metadata

    Args:
        NamedTuple (_type_): various tensorflow parameters
    """

    dataset_path: os.PathLike
    aligns_path: os.PathLike
    epochs: int
    frame_count: int
    image_width: int
    image_height: int
    image_channels: int
    max_string: int
    batch_size: int
    val_split: float
    use_cache: bool


def main():
    """Entry point of the script for training a model."""
    dataset_path = DATA_PATH
    aligns_path = ALIGN_PATH

    timestamp = datetime.utcnow().timestamp()

    config = TrainingConfig(
        dataset_path=dataset_path,
        aligns_path=aligns_path,
        epochs=int(os.environ.get("EPOCH", 60)),
        frame_count=int(os.environ.get("FRAME_COUNT", 75)),
        image_width=int(os.environ.get("IMAGE_WIDTH", 100)),
        image_height=int(os.environ.get("IMAGE_HEIGHT", 50)),
        image_channels=int(os.environ.get("IMAGE_CHANNELS", 3)),
        max_string=int(os.environ.get("MAX_STRING", 28)),
        batch_size=int(os.environ.get("BATCH_SIZE", 16)),
        val_split=float(os.environ.get("VAL_SPLIT", 0.2)),
        use_cache=bool(os.environ.get("USE_CACHE", True)),
    )

    train(timestamp, config)


def train(timestamp: float, config: TrainingConfig):
    """Training script

    Args:
        timestamp (float): current utc timestamp
        config (TrainingConfig): config class
    """

    logger = get_logger("lipnet_training")
    logger.info(
        "TRAINING: {}".format(
            datetime.utcfromtimestamp(timestamp).strftime(DATETIME_FMT)
        )
    )

    logger.info("For dataset at: {}".format(config.dataset_path))
    logger.info("With aligns at: {}".format(config.aligns_path))

    lipnet = LipNet(
        config.frame_count,
        config.image_channels,
        config.image_height,
        config.image_width,
        config.max_string,
    )
    lipnet.compile()

    datagen = DatasetGenerator(
        config.dataset_path,
        config.aligns_path,
        config.batch_size,
        config.max_string,
        config.val_split,
        config.use_cache,
    )

    callbacks = create_callbacks(timestamp, lipnet, datagen)

    logger.info("Starting training...")

    start_time = time.time()

    lipnet.fit(
        x=datagen.train_generator,
        validation_data=datagen.val_generator,
        epochs=config.epochs,
        verbose=1,
        shuffle=True,
        max_queue_size=5,
        workers=2,
        callbacks=callbacks,
        use_multiprocessing=True,
    )

    elapsed_time = time.time() - start_time
    print("Training completed in: {}".format(timedelta(seconds=elapsed_time)))


def create_callbacks(
    timestamp: float, lipnet: LipNet, datagen: DatasetGenerator
) -> list:
    """Create TF callback function list

    Args:
        timestamp (float): utc timestamp
        lipnet (LipNet): LipNet model
        datagen (DatasetGenerator): batch data generator

    Returns:
        list: list of tf.keras.callbacks
    """
    timestring = datetime.utcfromtimestamp(timestamp).strftime(DATETIME_FMT)

    # check log path
    batch_log_dir = LOG_PATH.joinpath(timestring)
    batch_log_dir.mkdir(exist_ok=True)

    # Tensorboard
    tensorboard = TensorBoard(LOG_PATH=str(batch_log_dir))

    # Model checkpoint saver
    checkpoint_dir = LIPNET_MODEL_SAVE_PATH.joinpath(timestring)
    checkpoint_dir.mkdir(exist_ok=True)
    checkpoint = ModelCheckpoint(
        filepath=checkpoint_dir.joinpath("lipnet_{epoch:03d}_{loss:.2f}.h5"),
        monitor="val_loss",
        save_weights_only=True,
        mode="auto",
        verbose=1,
    )

    return [checkpoint, tensorboard]


if __name__ == "__main__":
    main()
