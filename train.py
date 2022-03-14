import os
import time
import env
from typing import NamedTuple
from datetime import datetime
from colorama import Fore, init
from tensorflow.keras.callbacks import CSVLogger, ModelCheckpoint, TensorBoard

from common.decode import create_decoder
from core.callbacks.error_rates import ErrorRates
from core.generators.dataset_generator import DatasetGenerator
from core.model.lipnet import LipNet
from pathlib import Path

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
init(autoreset=True)


ROOT_PATH = Path(os.path.dirname(os.path.realpath(__file__))).resolve()
OUTPUT_DIR = Path(os.path.realpath(os.path.join(ROOT_PATH, "data", "res"))).resolve()
LOG_DIR = Path(os.path.realpath(os.path.join(ROOT_PATH, "data", "res_logs"))).resolve()

DICTIONARY_PATH = Path(
    os.path.realpath(
        ROOT_PATH.joinpath(os.path.join("data", "dictionaries", "grid.txt"))
    )
).resolve()

DATETIME_FMT = "%Y-%m-%d-%H-%M-%S"


class TrainingConfig(NamedTuple):
    dataset_path: str
    aligns_path: str
    epochs: int = 1
    frame_count: int = env.FRAME_COUNT
    image_width: int = env.IMAGE_WIDTH
    image_height: int = env.IMAGE_HEIGHT
    image_channels: int = env.IMAGE_CHANNELS
    max_string: int = env.MAX_STRING
    batch_size: int = env.BATCH_SIZE
    val_split: float = env.VAL_SPLIT
    use_cache: bool = True


def main():
    """
    Entry point of the script for training a model.
    i.e: python train.py -d data/dataset -a data/aligns -e 150
    """

    print(
        r"""
   __         __     ______   __   __     ______     ______  
  /\ \       /\ \   /\  == \ /\ "-.\ \   /\  ___\   /\__  _\ 
  \ \ \____  \ \ \  \ \  _-/ \ \ \-.  \  \ \  __\   \/_/\ \/ 
   \ \_____\  \ \_\  \ \_\    \ \_\\"\_\  \ \_____\    \ \_\ 
    \/_____/   \/_/   \/_/     \/_/ \/_/   \/_____/     \/_/ 

  implemented by Omar Salinas
	"""
    )

    # ap = argparse.ArgumentParser()

    # ap.add_argument('-d', '--dataset-path', required=True, help='Path to the dataset root directory')
    # ap.add_argument('-a', '--aligns-path', required=True, help='Path to the directory containing all align files')
    # ap.add_argument('-e', '--epochs', required=False, help='(Optional) Number of epochs to run', type=int, default=1)
    # ap.add_argument('-ic', '--ignore-cache', required=False, help='(Optional) Force the generator to ignore the cache file', action='store_true', default=False)

    # args = vars(ap.parse_args())

    # dataset_path = Path(os.path.realpath(args['dataset_path'])).resolve()
    # aligns_path  = Path(os.path.realpath(args['aligns_path'])).resolve()
    # epochs       = args['epochs']
    # ignore_cache = args['ignore_cache']
    dataset_path = Path(os.path.realpath("./data/dataset")).resolve()
    aligns_path = Path(os.path.realpath("./data/align")).resolve()
    epochs = 5
    ignore_cache = True

    if not dataset_path.is_dir():
        print(Fore.RED + "\nERROR: The dataset path is not a directory")
        return

    if not aligns_path.is_dir():
        print(Fore.RED + "\nERROR: The aligns path is not a directory")
        return

    if not isinstance(epochs, int) or epochs <= 0:
        print(
            Fore.RED
            + "\nERROR: The number of epochs must be a valid integer greater than zero"
        )
        return

    timestamp = datetime.utcnow().timestamp()
    config = TrainingConfig(
        dataset_path, aligns_path, epochs=epochs, use_cache=not ignore_cache
    )

    train(timestamp, config)


def train(timestamp: str, config: TrainingConfig):
    print(
        "\nTRAINING: {}\n".format(
            datetime.utcfromtimestamp(timestamp).strftime(DATETIME_FMT)
        )
    )

    print("For dataset at: {}".format(config.dataset_path))
    print("With aligns at: {}".format(config.aligns_path))

    OUTPUT_DIR.mkdir(exist_ok=True)
    LOG_DIR.mkdir(exist_ok=True)

    lipnet = LipNet(
        config.frame_count,
        config.image_channels,
        config.image_height,
        config.image_width,
        config.max_string,
    ).compile_model()

    datagen = DatasetGenerator(
        config.dataset_path,
        config.aligns_path,
        config.batch_size,
        config.max_string,
        config.val_split,
        config.use_cache,
    )

    # callbacks = create_callbacks(timestamp, lipnet, datagen)

    print("\nStarting training...\n")

    start_time = time.time()

    lipnet.model.fit(
        x=datagen.train_generator,
        validation_data=datagen.val_generator,
        epochs=config.epochs,
        verbose=1,
        shuffle=True,
        max_queue_size=5,
        workers=2,
        # callbacks=callbacks,
        use_multiprocessing=True,
    )

    elapsed_time = time.time() - start_time
    print(
        "\nTraining completed in: {}".format(datetime.timedelta(seconds=elapsed_time))
    )


def create_callbacks(timestamp: str, lipnet: LipNet, datagen: DatasetGenerator) -> list:
    timestring = datetime.utcfromtimestamp(timestamp).strftime(DATETIME_FMT)
    run_log_dir = LOG_DIR.joinpath(timestring)

    run_log_dir.mkdir(exist_ok=True)

    # Tensorboard
    tensorboard = TensorBoard(log_dir=run_log_dir.as_posix())

    # Training logger
    csv_log = run_log_dir.joinpath("training.csv")
    csv_logger = CSVLogger(csv_log, separator=",", append=True)

    # Model checkpoint saver
    checkpoint_dir = OUTPUT_DIR.joinpath(timestring)

    checkpoint_dir.mkdir(exist_ok=True)

    checkpoint_template = checkpoint_dir.joinpath("lipnet_{epoch:03d}_{loss:.2f}.h5")

    checkpoint = ModelCheckpoint(
        checkpoint_template,
        monitor="val_loss",
        save_weights_only=True,
        mode="auto",
        save_freq=1,
        verbose=1,
    )

    # WER/CER Error rate calculator
    error_rate_log = run_log_dir.joinpath("error_rates.csv")

    decoder = create_decoder(DICTIONARY_PATH, False)
    error_rates = ErrorRates(error_rate_log, lipnet, datagen.val_generator, decoder)

    return [checkpoint, csv_logger, error_rates, tensorboard]


if __name__ == "__main__":
    main()
