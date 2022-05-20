import os
import sys
from pathlib import Path
from typing import NamedTuple
from datetime import datetime
from dotenv import load_dotenv

root = Path(__file__).parents[2].resolve()
sys.path.insert(0, str(root))

assert load_dotenv(".env")

from utils.logger import get_logger
from core.callbacks.callbacks import (
    ModelCheckpoint,
    TensorBoard,
    Callback,
)
from core.generators.lipnet_dataset import DatasetGenerator
from core.models.lipnet import LipNet
from core.utils.config import LipNetConfig
from core.utils.types import Path, List

# define dataset paths
DATA_PATH = root.joinpath(root.joinpath("data/grid/npy"))
ALIGN_PATH = root.joinpath(root.joinpath("data/grid/align"))

# define training params
LIPNET_MODEL_SAVE_PATH = root.joinpath("models/lipnet")
LOG_PATH = root.joinpath("logs/lipnet")

DICTIONARY_PATH = root.joinpath("data/dictionaries/grid.txt")

LIPNET_MODEL_SAVE_PATH.mkdir(exist_ok=True)
LOG_PATH.mkdir(exist_ok=True)

DATETIME_FMT = os.environ["DATETIME_FMT"]


class TrainingConfig(NamedTuple):
    """Baisc Training Metadata

    Args:
        NamedTuple (_type_): various tensorflow parameters
    """

    dataset_path: Path
    aligns_path: Path
    epochs: int
    frame_count: int
    image_width: int
    image_height: int
    image_channels: int
    max_string: int
    batch_size: int
    val_split: float
    use_cache: bool


def train(timestring: str, config: TrainingConfig):
    """Training script

    Args:
        timestamp (float): current utc timestamp
        config (TrainingConfig): config class
    """

    logger = get_logger("LipNet")

    logger.info(f"Time: {timestring}")

    logger.info(f"Dataset: {config.dataset_path}")
    logger.info(f"Aligns: {config.aligns_path}")

    lipnet_config = LipNetConfig(
        frame_count=config.frame_count,
        image_height=config.image_height,
        image_width=config.image_width,
        image_channels=config.image_channels,
        max_string=config.max_string,
        output_size=28,
    )
    lipnet = LipNet(lipnet_config)
    lipnet.compile()

    datagen = DatasetGenerator(
        config.dataset_path,
        config.aligns_path,
        config.batch_size,
        config.max_string,
        config.val_split,
        config.use_cache,
    )

    callbacks = create_callbacks(timestring)

    logger.info("Start training")

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


def create_callbacks(timestring: str) -> List[Callback]:
    """Create TF callback function list

    Args:
        timestring (str): utc timestring

    Returns:
        list: list of tf.keras.callbacks
    """

    # check log path
    batch_log_dir = LOG_PATH.joinpath(timestring)
    batch_log_dir.mkdir(exist_ok=True)

    # Tensorboard
    tensorboard = TensorBoard(log_dir=str(batch_log_dir))

    # Model checkpoint saver
    checkpoint_dir = LIPNET_MODEL_SAVE_PATH.joinpath(timestring)
    checkpoint_dir.mkdir(exist_ok=True)
    checkpoint = ModelCheckpoint(
        filepath=checkpoint_dir.joinpath("lipnet_{epoch:02d}_{val_loss:.2f}.h5"),
        monitor="val_loss",
        save_weights_only=True,
        mode="auto",
        verbose=1,
    )

    return [checkpoint, tensorboard]


if __name__ == "__main__":
    dataset_path = DATA_PATH
    aligns_path = ALIGN_PATH

    timestring = datetime.utcnow().strftime(DATETIME_FMT)

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

    train(timestring, config)
