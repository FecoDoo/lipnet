import os
import sys
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv

root = Path(__file__).parents[2].resolve()
sys.path.insert(0, str(root))

assert load_dotenv(".env")

from core.utils.config import BaselineConfig, emotion
from core.utils.types import PathLike, Optional

from tensorflow import dtypes, one_hot, io, image, data
from tensorflow.keras.callbacks import (
    ReduceLROnPlateau,
    ModelCheckpoint,
    EarlyStopping,
    TensorBoard,
)
from core.models.baseline import DenseNet
from core.utils.config import BaselineConfig


training_timestamp = datetime.utcnow().strftime(os.environ["DATETIME_FMT"])

data_dir = root.joinpath("data/affectnet")
log_dir = root.joinpath(os.path.join("logs/baseline/", training_timestamp))
model_save_dir = root.joinpath(os.path.join("models/baseline", training_timestamp))

assert data_dir.exists()
log_dir.mkdir(exist_ok=True)
model_save_dir.mkdir(exist_ok=True)

# training params
batch_size = 16
epochs = 40  # 训练轮数

# model params
densenet_config = BaselineConfig()

frame_shape = (
    densenet_config.image_height,
    densenet_config.image_width,
    densenet_config.image_channels,
)


def process_image(path):
    """generate tensorflow.data.Dataset from images

    Args:
        path (Tensor): Tensor object from tensorflow.data.Dataset

    Returns:
        img: tensorflow image object
    """
    # load image
    img = io.read_file(path)
    img = io.decode_jpeg(img, channels=3)
    img = image.convert_image_dtype(img, dtypes.float32)
    img = image.resize(img, frame_shape[:2])
    img = img / 255.0

    return img


def generate_tf_dataset(path: os.PathLike):

    image_list = list(path.rglob("*.jpg"))

    label_list = [emotion[path.parent.name] for path in image_list]

    # image
    image_dataset = data.Dataset.from_tensor_slices(
        [str(path) for path in image_list]
    ).map(process_image)

    # label
    label_dataset = data.Dataset.from_tensor_slices(label_list).map(
        lambda x: one_hot(indices=x, depth=7, dtype=dtypes.uint8)
    )

    return data.Dataset.zip((image_dataset, label_dataset))


def generate_callbacks() -> list:
    """generate tensorflow callback functions

    Returns:
        list: list of tensorflow callback instance
    """
    # set learning rate reducing policy
    reduce_learning_rate = ReduceLROnPlateau(
        monitor="val_loss", factor=0.2, patience=5, verbose=0
    )  # 学习率衰减策略

    # set checkpoints
    checkpoint = ModelCheckpoint(
        model_save_dir.joinpath("densenet121_{epoch:02d}_{val_loss:.2f}.h5"),
        monitor="val_loss",
        verbose=1,
        save_best_only=True,
        save_weights_only=True,
        mode="auto",
        save_freq="epoch",
    )

    # early stopping strategy
    early_stopping = EarlyStopping(
        monitor="val_loss", patience=4, verbose=0, mode="auto"
    )
    # Tensorboard
    tensorboard = TensorBoard(log_dir=str(log_dir))

    return [checkpoint, reduce_learning_rate, tensorboard]


def load_baseline(pivot=313):
    """load pre-trained densenet121 and set learnable layers

    Returns:
        model: TensorFlow Model instance
    """
    model = DenseNet(BaselineConfig())

    # for layer in model.basemodel.layers[:pivot]:
    #     layer.trainable = False
    # for layer in model.basemodel.layers[pivot:]:
    #     layer.trainable = True

    return model


def start_training(weights: Optional[PathLike]):
    """main training loop

    Args:
        weights (Optional[PathLike]): baseline model weight filepath
    """

    # generate dataset and preprocessing

    train_dataset = (
        generate_tf_dataset(data_dir.joinpath("train/faces"))
        .batch(batch_size=batch_size, drop_remainder=False)
        .prefetch(buffer_size=2)
    )
    test_dataset = (
        generate_tf_dataset(data_dir.joinpath("test/faces"))
        .batch(batch_size=batch_size, drop_remainder=True)
        .prefetch(buffer_size=2)
    )

    model = load_baseline()

    # fine-tuning
    if weights is not None:
        model.load_weights(weights)
        model.basemodel.trainable = True
        model.compile(adam_learning_rate=1e-5)

    else:
        model.compile(adam_learning_rate=1e-3)

    callbacks = generate_callbacks()

    history = model.fit(
        x=train_dataset,
        epochs=epochs,
        callbacks=callbacks,
        validation_data=test_dataset,
    )


if __name__ == "__main__":
    start_training(weights=None)
