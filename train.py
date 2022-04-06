import os
import time
from typing import NamedTuple
from datetime import datetime, timedelta
from pathlib import Path

from common.decode import create_decoder
from core.callbacks.callbacks import ErrorRates, CSVLogger, ModelCheckpoint, TensorBoard
from core.generators.dataset_generator import DatasetGenerator
from core.model.lipnet import LipNet
import env

os.environ["TF_CPP_MIN_LOG_LEVEL"] = env.TF_CPP_MIN_LOG_LEVEL


ROOT_PATH = Path(os.path.dirname(os.path.realpath(__file__))).resolve()
DATA_PATH = ROOT_PATH.joinpath("data/dataset")
ALIGN_PATH = ROOT_PATH.joinpath("data/align")

assert DATA_PATH.is_dir() and ALIGN_PATH.is_dir()

MODEL_DIR = ROOT_PATH.joinpath("models")
LOG_DIR = ROOT_PATH.joinpath("logs/training")
DICTIONARY_PATH = ROOT_PATH.joinpath("data/dictionaries/grid.txt")

MODEL_DIR.mkdir(exist_ok=True)
LOG_DIR.mkdir(exist_ok=True)

DATETIME_FMT = "%Y-%m-%d-%H-%M-%S"


class TrainingConfig(NamedTuple):
    """Baisc Training Metadata

    Args:
        NamedTuple (_type_): various tensorflow parameters
    """

    dataset_path: os.PathLike
    aligns_path: os.PathLike
    epochs: int = env.EPOCH
    frame_count: int = env.FRAME_COUNT
    image_width: int = env.IMAGE_WIDTH
    image_height: int = env.IMAGE_HEIGHT
    image_channels: int = env.IMAGE_CHANNELS
    max_string: int = env.MAX_STRING
    batch_size: int = env.BATCH_SIZE
    val_split: float = env.VAL_SPLIT
    use_cache: bool = env.USE_CACHE


def main():
    """Entry point of the script for training a model.
    i.e: python train.py -d data/dataset -a data/aligns -e 150
    """
    dataset_path = DATA_PATH
    aligns_path = ALIGN_PATH

    timestamp = datetime.utcnow().timestamp()

    config = TrainingConfig(dataset_path, aligns_path)

    train(timestamp, config)


def train(timestamp: float, config: TrainingConfig):
    """Training script

    Args:
        timestamp (float): current utc timestamp
        config (TrainingConfig): config class
    """
    print(
        "\nTRAINING: {}\n".format(
            datetime.utcfromtimestamp(timestamp).strftime(DATETIME_FMT)
        )
    )

    print("For dataset at: {}".format(config.dataset_path))
    print("With aligns at: {}".format(config.aligns_path))

    MODEL_DIR.mkdir(exist_ok=True)
    LOG_DIR.mkdir(exist_ok=True)

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

    print("\nStarting training...\n")

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
    print("\nTraining completed in: {}".format(timedelta(seconds=elapsed_time)))


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
    batch_log_dir = LOG_DIR.joinpath(timestring)
    batch_log_dir.mkdir(exist_ok=True)

    # Tensorboard
    tensorboard = TensorBoard(log_dir=str(batch_log_dir))

    # Training logger
    csv_log_path = batch_log_dir.joinpath("training.csv")
    csv_logger = CSVLogger(str(csv_log_path), separator=",", append=True)

    # Model checkpoint saver
    checkpoint_dir = MODEL_DIR.joinpath(timestring)
    checkpoint_dir.mkdir(exist_ok=True)
    checkpoint = ModelCheckpoint(
        filepath=checkpoint_dir.joinpath("lipnet_{epoch:03d}_{loss:.2f}.h5"),
        monitor="val_loss",
        save_weights_only=True,
        mode="auto",
        verbose=1,
    )

    # WER/CER Error rate calculator
    error_rate_log = batch_log_dir.joinpath("error_rates.csv")

    decoder = create_decoder(DICTIONARY_PATH, False)
    error_rates = ErrorRates(error_rate_log, lipnet, datagen.val_generator, decoder)

    return [checkpoint, tensorboard]


if __name__ == "__main__":
    main()
