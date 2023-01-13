import numpy as np
import pandas as pd
import tensorflow as tf
import core.generators.baseline_dataset as baseline_dataset
import core.generators.dnn_dataset as dnn_dataset
from core.utils.config import emotion


def normalization(mat):
    min_v, max_v = np.min(mat), np.max(mat)
    return (mat - min_v) / (max_v - min_v)


def predicting(model, dataset, num_frames=75, flag="baseline"):
    """Prediction per round

    Args:
        model (_type_): _description_
        dataset (_type_): _description_
        num_frames (int, optional): _description_. Defaults to 75.
        flag (str, optional): _description_. Defaults to "baseline".

    Returns:
        _type_: _description_
    """
    preds = tf.concat(
        [
            model.predict(dataset.train),
            model.predict(dataset.validation),
        ],
        axis=0,
    )

    if flag == "baseline":
        # need to repeat labels as baseline model output one prediction for each image in the batch
        labels = tf.concat(
            [   
                tf.one_hot(
                    indices=np.repeat(
                        [
                            emotion[sample.parents[0].name]
                            for sample in dataset.train.face_paths
                        ],
                        num_frames
                    ),
                    depth=7,
                ),
                tf.one_hot(
                    indices=np.repeat(
                        [
                            emotion[sample.parents[0].name]
                            for sample in dataset.validation.face_paths
                        ],
                        num_frames
                    ),
                    depth=7,
                ),
            ],
            axis=0,
        )
    else:
        labels = tf.concat(
            [
                tf.one_hot(
                    [
                        emotion[sample.parents[0].name]
                        for sample in dataset.train.face_paths
                    ],
                    depth=7,
                ),
                tf.one_hot(
                    [
                        emotion[sample.parents[0].name]
                        for sample in dataset.validation.face_paths
                    ],
                    depth=7,
                ),
            ],
            axis=0,
        )

    return preds, labels


def batch_predicting(model, dataset_path, num_frames=75, val_split=0.2, n_loop=60, flag="baseline"):
    """Predicting facial expressions from images/videos 

    Args:
        model (tf.model): TensorFlow Model (compiled)
        dataset_path (os.PathLike): relative path to the dataset
        num_frames (int, optional): number of frames in a batch. Defaults to 75.
        val_split (float, optional): train/test set split ratio. Defaults to 0.2.
        n_loop (int, optional): n rounds of evaluation. Defaults to 60.
        flag (str, optional): choosing model type. Defaults to "baseline".

    Raises:
        AttributeError: _description_
        AttributeError: _description_

    Returns:
        (tf.Tensor, tf.Tensor): predictions and ground-true labels
    """
    if model is None or dataset_path is None:
        raise AttributeError("parameter missing")

    predictions, labels = None, None

    print("generating dataset")

    if flag == "baseline":
        dataset = baseline_dataset.DatasetGenerator(
            dataset_path=dataset_path,
            num_frames=num_frames,
            val_split=val_split
        )
    elif flag == "dnn":
        dataset = dnn_dataset.DatasetGenerator(
            dataset_path=dataset_path,
            num_frames=num_frames,
            val_split=val_split
        )
    else:
        raise AttributeError("only dnn/baseline generatora are suppoted")

    for i in range(n_loop):
        print(f"prediction loop {i}")

        temp_preds, temp_labels = predicting(model, dataset, num_frames, flag)

        if i == 0:
            predictions, labels = temp_preds, temp_labels
        else:
            predictions = tf.concat(
                [
                    predictions,
                    temp_preds
                ],
                axis=0,
            )

            labels = tf.concat(
                [
                    labels,
                    temp_labels
                ],
                axis=0,
            )

    return predictions, labels
