# %%
import numpy as np
import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from pathlib import Path
from skvideo.io import vread

preprocessing = False
project_dir = Path(__file__).parent.resolve()
log_dir = project_dir.joinpath("logs/baseline/training")
model_dir = project_dir.joinpath("models/baseline")

batch_size = 8
# %% [markdown]
# # Info

# %%
info = pd.read_csv(project_dir.joinpath("data/expw/info.csv"), header=0)

# %% [markdown]
# # Cropping faces

# %%
if preprocessing:

    from PIL import Image

    ROOT = Path("/l/expw/")

    from pandarallel import pandarallel

    pandarallel.initialize(progress_bar=True)

    emotion_dict = {}

    def process(image_name):
        file_path = ROOT.joinpath("image", image_name)
        record = info[info["image_name"] == image_name].to_numpy()

        for face in record:
            label = face[-1]
            dst = ROOT.joinpath(
                "faces",
                str(label)
                + "_"
                + image_name.split(".")[0]
                + "_"
                + str(face[1])
                + ".jpg",
            )

            if os.path.exists(dst):
                continue

            img = Image.open(file_path).crop(box=(face[3], face[2], face[4], face[5]))

            img.save(dst)

    _ = info["image_name"].parallel_apply(process)

else:
    print("skip")


# %% [markdown]
# # Generate TF Dataset

# %%
dataset = tf.data.Dataset.list_files("/l/expw/faces/*.jpg")

# %%
def preprocessing_images(path):

    label = tf.strings.to_number(
        tf.strings.split(
            tf.strings.split(tf.strings.split(path, os.sep)[-1], ".")[0], "_"
        )[0],
        out_type=tf.dtypes.int32,
    )

    label = tf.one_hot(tf.cast(label, tf.dtypes.int32), 7)

    img = tf.io.read_file(path)
    img = tf.io.decode_jpeg(img)
    img = tf.image.convert_image_dtype(img, tf.dtypes.float32)
    # img = tf.image.pad_to_bounding_box(
    #     img, 0, 0, target_height, target_width
    # )

    img = tf.image.resize(img, [224, 224])
    return img, label


# %%
dataset = dataset.map(preprocessing_images)

# %% [markdown]
# # Batching

# %%
batched_dataset = (
    dataset.shuffle(buffer_size=1600).batch(batch_size=batch_size, drop_remainder=False).prefetch(buffer_size=2)
)

# Prepare the validation dataset
val_dataset = (
    dataset.shuffle(buffer_size=1600).batch(batch_size=batch_size, drop_remainder=False).prefetch(buffer_size=2)
)

# %% [markdown]
# # Model

# %%
from tensorflow.keras.applications.densenet import DenseNet121
from tensorflow.keras.applications.densenet import preprocess_input
from tensorflow.keras.models import Model

baseline = DenseNet121(
    weights="imagenet", include_top=False, input_shape=(224, 224, 3), pooling="avg"
)

# %%
for layer in baseline.layers[:149]:
    layer.trainable = False
for layer in baseline.layers[149:]:
    layer.trainable = True

# %% [markdown]
# # Training

# %%
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

# %%
learning_rate = 1e-4  # 设置学习率为1e-4
epochs = 5  # 训练轮数

# %%
reduce_learning_rate = ReduceLROnPlateau(
    monitor="val_loss", factor=0.1, patience=2, verbose=1
)  # 学习率衰减策略
# 断点训练，有利于恢复和保存模型
checkpoint = ModelCheckpoint(
    model_dir.joinpath("densenet121_{epoch:03d}_{loss:.2f}.h5"),
    monitor="val_accuracy",
    verbose=1,
    save_best_only=True,
    save_weights_only=True,
    mode="auto",
    save_freq='epoch',
)
# early stopping策略
early_stopping = EarlyStopping(
    monitor="val_accuracy", patience=4, verbose=1, mode="auto"
)
# Tensorboard
tensorboard = TensorBoard(log_dir=str(log_dir))

# %%
baseline.output.shape

# %%

layer = Dense(256, activation="relu")(baseline.output)
layer = Dropout(0.7)(layer)
layer = Dense(128, activation="relu")(layer)
layer = Dropout(0.5)(layer)
layer = Dense(64, activation="relu")(layer)
layer = Dropout(0.3)(layer)
output = Dense(7, activation="softmax")(layer)

model = Model(inputs=baseline.input, outputs=output)

model.compile(
    loss=categorical_crossentropy,
    optimizer=Adam(learning_rate=learning_rate),
    metrics=["accuracy"],
)

model.load_weights(model_dir.joinpath("densenet121.h5"))

# %%
history = model.fit(
    x=batched_dataset,
    epochs=epochs,
    callbacks=[early_stopping, checkpoint, reduce_learning_rate],
    validation_data=val_dataset,
)
