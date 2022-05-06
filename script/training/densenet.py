# %%
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Tuple
from dotenv import load_dotenv

root = Path(__file__).parents[2].resolve()
sys.path.insert(0, str(root))

assert load_dotenv(".env")

from core.utils.config import BaselineConfig, emotion
from core.utils.types import PathLike, Optional

from tensorflow import dtypes, one_hot, io, image, data
from tensorflow.keras.applications.densenet import DenseNet121
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Dense,
    Dropout,
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import (
    ReduceLROnPlateau,
    ModelCheckpoint,
    EarlyStopping,
    TensorBoard,
)
from tensorflow.keras.losses import categorical_crossentropy


training_timestamp = datetime.utcnow().strftime(os.environ['DATETIME_FMT'])

data_dir = root.joinpath("data/affectnet")
log_dir = root.joinpath(os.path.join("logs/baseline/", training_timestamp))
model_save_dir = root.joinpath(os.path.join("models/baseline", training_timestamp))

assert data_dir.exists()
log_dir.mkdir(mode=750, exist_ok=True)
model_save_dir.mkdir(mode=750, exist_ok=True)

# training params
batch_size = 32
learning_rate = 1e-3  # 设置学习率为1e-3
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
        monitor="val_accuracy", factor=0.2, patience=5, verbose=0
    )  # 学习率衰减策略

    # set checkpoints
    checkpoint = ModelCheckpoint(
        model_save_dir.joinpath(
            "densenet121_{epoch:03d}_{val_accuracy:.2f}_{val_loss:.2f}.h5"
        ),
        monitor="val_accuracy",
        verbose=1,
        save_best_only=True,
        save_weights_only=True,
        mode="auto",
        save_freq="epoch",
    )

    # early stopping strategy
    early_stopping = EarlyStopping(
        monitor="val_accuracy", patience=4, verbose=0, mode="auto"
    )
    # Tensorboard
    tensorboard = TensorBoard(log_dir=str(log_dir))

    return [early_stopping, checkpoint, reduce_learning_rate, tensorboard]


def load_densenet():
    """load pre-trained densenet121 and set learnable layers

    Returns:
        densenet: DenseNet121 instance
    """
    densenet = DenseNet121(
        weights="imagenet", include_top=False, input_shape=frame_shape, pooling="avg"
    )

    for layer in densenet.layers[:149]:
        layer.trainable = False
    for layer in densenet.layers[149:]:
        layer.trainable = True

    return densenet


def customize_layers(baseline_model):
    """add layers to baseline model

    Args:
        baseline_model (DenseNet121): baseline model instance

    Returns:
        layer: last layer of the new model
    """
    layer = Dense(256, activation="relu")(baseline_model.output)
    layer = Dropout(0.7)(layer)
    layer = Dense(128, activation="relu")(layer)
    layer = Dropout(0.5)(layer)
    layer = Dense(64, activation="relu")(layer)
    layer = Dropout(0.3)(layer)
    layer = Dense(7, activation="softmax")(layer)

    return layer


def start_training(weights: Optional[PathLike]):
    """main training loop

    Args:
        weights (Optional[PathLike]): baseline model weight filepath
    """

    # generate dataset and preprocessing

    train_dataset = (
        generate_tf_dataset(data_dir.joinpath("train/faces"))
        .shuffle(buffer_size=1000)
        .batch(batch_size=batch_size, drop_remainder=False)
        .prefetch(buffer_size=2)
    )
    test_dataset = (
        generate_tf_dataset(data_dir.joinpath("test/faces"))
        .shuffle(buffer_size=1000)
    )

    densenet = load_densenet()

    output = customize_layers(densenet)

    model = Model(inputs=densenet.input, outputs=output)

    model.compile(
        loss=categorical_crossentropy,
        optimizer=Adam(learning_rate=learning_rate),
        metrics=["accuracy"],
    )

    if weights:
        model.load_weights(weights)

    callbacks = generate_callbacks()

    history = model.fit(
        x=train_dataset,
        epochs=epochs,
        callbacks=callbacks,
        validation_data=test_dataset,
    )


if __name__ == "__main__":
    start_training(weights=None)
