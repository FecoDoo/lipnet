import os
import sys
import numpy as np
from dotenv import load_dotenv
from datetime import datetime
from pathlib import Path
from tensorflow.keras.callbacks import (
    ReduceLROnPlateau,
    ModelCheckpoint,
    EarlyStopping,
    TensorBoard,
)

root = Path(__file__).parents[2].resolve()
sys.path.insert(0, str(root))

assert load_dotenv(".env")

training_timestamp = datetime.utcnow().strftime(os.environ["DATETIME_FMT"])
data_dir = root.joinpath("data/ravdess/videos")

batch_size = 4
epochs = 20  # 训练轮数

assert data_dir.exists()

from core.models.dnn import DNN
from core.utils.config import LipNetConfig, BaselineConfig

model = DNN(
    lipnet_config=LipNetConfig(),
    baseline_config=BaselineConfig(),
    lipnet_model_weight=root.joinpath("models/lipnet/lipnet.h5"),
    baseline_model_weight=root.joinpath("models/baseline/mobilenet"),
)

model.compile(learning_rate=1e-3, metrics=["accuracy"])

# paths
log_dir = root.joinpath(os.path.join("logs/dnn/", training_timestamp))
model_save_dir = root.joinpath(os.path.join("models/dnn", training_timestamp))

model_save_dir.mkdir(exist_ok=True)
log_dir.mkdir(exist_ok=True)

# create generator
from core.generators.dnn_generator import VideoSampleGenerator

video_paths = list(data_dir.rglob("*.mp4"))

n_videos = len(video_paths)
split_pivot = int(np.floor(n_videos * 0.8))

generator_train = VideoSampleGenerator(
    video_paths=video_paths[:split_pivot], batch_size=batch_size
)
generator_test = VideoSampleGenerator(
    video_paths=video_paths[split_pivot:], batch_size=batch_size
)


def generate_callbacks(model_name: str) -> list:
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
        model_save_dir.joinpath(model_name + "_{epoch:02d}_{val_loss:.2f}.h5"),
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


if __name__ == "__main__":
    callbacks = generate_callbacks("dnn")

    history = model.fit(
        x=generator_train,
        epochs=20,
        callbacks=callbacks,
        validation_data=generator_test,
    )
