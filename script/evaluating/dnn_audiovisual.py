# %%
import os
import sys
import numpy as np
import pandas as pd
import tensorflow as tf
from dotenv import load_dotenv
from datetime import datetime
from typing import List
from pathlib import Path

root = Path("/m/cs/work/kaiy1/lipnet").resolve()
sys.path.insert(0, str(root))

assert load_dotenv(".env")

# %%
dataset_name = "casia"

# init logger
from utils.logger import get_logger
logger = get_logger(f"DNN_{dataset_name}_evaluation")

training_timestamp = datetime.utcnow().strftime(os.environ["DATETIME_FMT"])

dataset_path = root.joinpath(f"data/{dataset_name}")
assert dataset_path.exists()

# %%
from core.models.dnn import DNN
from core.utils.config import LipNetConfig, BaselineConfig

lipnet_weight = root.joinpath("models/lipnet/lipnet.h5")
baseline_weight = root.joinpath("models/baseline/mobilenet")

model = DNN(LipNetConfig(), BaselineConfig(), lipnet_weight, baseline_weight)
model.compile(metrics=['accuracy'])

# %%
# create generator
import tensorflow_addons as tfa
from core.utils.config import emotion
from common.evaluator import batch_predicting

metric = tfa.metrics.F1Score(num_classes=7, threshold=0.5, average="macro")
metric.reset_state()

logger.info("calculating confusion matrix...")

# continous
model.load_weights(root.joinpath("models/dnn/2022-08-24T1033/dnn_67_0.29.h5"))

p, l = batch_predicting(
    model=model,
    dataset_path=dataset_path,
    val_split=0.2,
    n_loop=2,
    num_frames=75,
    flag="dnn"
)

metric.update_state(l, p)
c = metric.result().numpy()

metric.reset_state()
# random
model.load_weights(root.joinpath("models/dnn/2022-08-31T1401/dnn_59_0.09.h5"))

p, l = batch_predicting(
    model=model,
    dataset_path=dataset_path,
    val_split=0.2,
    n_loop=2,
    num_frames=75,
    flag="dnn"
)

metric.update_state(l, p)
r = metric.result().numpy()


# %%
logger.info(f"continuous draw F1 score: {c}")
logger.info(f"random draw F1 score: {r}")


