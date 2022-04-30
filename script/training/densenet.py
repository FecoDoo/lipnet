# %%
import os
import sys
from datetime import datetime
from pathlib import Path

root = Path(__file__).parents[2].resolve()
sys.path.insert(0, str(root))

from core.utils.config import DenseNetConfig, emotion
from core.utils.types import PathLike, Optional

from tensorflow import dtypes, one_hot, io, image, cast, data, strings
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


timestamp = datetime.utcnow().strftime("%Y-%m-%d-%H%M")

data_dir = root.joinpath("data/expw")
log_dir = root.joinpath("logs/baseline/training", timestamp)
model_save_dir = root.joinpath("models/baseline", timestamp)

assert data_dir.exists()
log_dir.mkdir(mode=750, exist_ok=True)
model_save_dir.mkdir(mode=750, exist_ok=True)

# training params
batch_size = 8
learning_rate = 1e-3  # 设置学习率为1e-4
epochs = 40  # 训练轮数

densenet_config = DenseNetConfig()

frame_shape = (
    densenet_config.image_width,
    densenet_config.image_height,
    densenet_config.image_channels,
)


def generate_tf_dataset(path):

    label = strings.to_number(
        strings.split(strings.split(strings.split(path, os.sep)[-1], ".")[0], "_")[0],
        out_type=dtypes.int32,
    )

    label = one_hot(indices=label, depth=7, dtype=dtypes.int32)

    img = io.read_file(path)
    img = io.decode_jpeg(img)
    img = image.convert_image_dtype(img, dtypes.float32)
    # img = image.pad_to_bounding_box(
    #     img, 0, 0, target_height, target_width
    # )

    img = image.resize(img, frame_shape[:2])

    return img, label


def generate_callbacks():
    # set learning rate reducing policy
    reduce_learning_rate = ReduceLROnPlateau(
        monitor="val_loss", factor=0.1, patience=2, verbose=0
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
    densenet = DenseNet121(
        weights="imagenet", include_top=False, input_shape=frame_shape, pooling="avg"
    )

    for layer in densenet.layers[:149]:
        layer.trainable = False
    for layer in densenet.layers[149:]:
        layer.trainable = True

    return densenet


def customize_layers(baseline_model):
    layer = Dense(256, activation="relu")(baseline_model.output)
    layer = Dropout(0.7)(layer)
    layer = Dense(128, activation="relu")(layer)
    layer = Dropout(0.5)(layer)
    layer = Dense(64, activation="relu")(layer)
    layer = Dropout(0.3)(layer)
    layer = Dense(7, activation="softmax")(layer)

    return layer


def start_training(weights: Optional[PathLike]):

    dataset = data.Dataset.list_files(str(data_dir.joinpath("face/*.jpg"))).map(
        generate_tf_dataset
    )

    batched_dataset = (
        dataset.shuffle(buffer_size=1600)
        .batch(batch_size=batch_size, drop_remainder=False)
        .prefetch(buffer_size=2)
    )

    # Prepare the validation dataset
    val_dataset = (
        dataset.shuffle(buffer_size=1600)
        .batch(batch_size=batch_size, drop_remainder=False)
        .prefetch(buffer_size=2)
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
        x=batched_dataset,
        epochs=epochs,
        callbacks=callbacks,
        validation_data=val_dataset,
    )


if __name__ == "__main__":
    start_training(weights=None)
