from tensorflow.keras import backend as k
from tensorflow.keras.layers import Input, Concatenate
from tensorflow.keras.layers import Conv3D, ZeroPadding3D
from tensorflow.keras.layers import (
    Activation,
    Dense,
    Flatten,
    Lambda,
    SpatialDropout3D,
    Dropout,
)
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import MaxPooling3D
from tensorflow.keras.layers import GRU
from tensorflow.keras.layers import Bidirectional, TimeDistributed
from typing import List


INPUT_TYPE = "float32"

ZERO_PADDING = (1, 2, 2)

ACTIVATION_FN = "relu"

CONV_KERNEL_INIT = "he_normal"
CONV_KERNEL_SIZE = (3, 5, 5)
CONV_STRIDES = (1, 2, 2)

POOL_SIZE = (1, 2, 2)
POOL_STRIDES = (1, 2, 2)

DROPOUT_RATE = 0.5

GRU_ACTIVATION = "tanh"
GRU_UNITS = 256
GRU_KERNEL_INIT = "Orthogonal"
GRU_MERGE_MODE = "concat"


def create_input_layer(name: str, shape, dtype: str = INPUT_TYPE):
    return Input(shape=shape, dtype=dtype, name=name)


def create_zero_layer(
    name: str, input_layer, padding: tuple = ZERO_PADDING
) -> ZeroPadding3D:
    return ZeroPadding3D(padding=padding, name=name)(input_layer)


def create_conv_layer(
    name: str, input_layer, filters: int, kernel_size: tuple = CONV_KERNEL_SIZE
) -> Conv3D:
    return Conv3D(
        filters=filters,
        kernel_size=kernel_size,
        strides=CONV_STRIDES,
        kernel_initializer=CONV_KERNEL_INIT,
        name=name,
    )(input_layer)


def create_batc_layer(name: str, input_layer) -> BatchNormalization:
    return BatchNormalization(name=name)(input_layer)


def create_softmax_layer(
    name: str, input_layer, activation: str = "softmax", units=7
) -> Dense:
    return Dense(units=units, activation=activation, name=name)(input_layer)


def create_actv_layer(
    name: str, input_layer, activation: str = ACTIVATION_FN
) -> Activation:
    return Activation(activation, name=name)(input_layer)


def create_pool_layer(name: str, input_layer) -> MaxPooling3D:
    return MaxPooling3D(pool_size=POOL_SIZE, strides=POOL_STRIDES, name=name)(
        input_layer
    )


def create_drop_layer(name: str, input_layer, rate=0) -> SpatialDropout3D:
    return Dropout(rate=rate, name=name)(input_layer)


def create_spatial_drop_layer(name: str, input_layer) -> SpatialDropout3D:
    return SpatialDropout3D(DROPOUT_RATE, name=name)(input_layer)


def create_bi_gru_layer(
    name: str, input_layer, units: int = GRU_UNITS, activation: str = GRU_ACTIVATION
) -> Bidirectional:
    return Bidirectional(
        GRU(
            units,
            return_sequences=True,
            activation=activation,
            kernel_initializer=GRU_KERNEL_INIT,
            name=name,
        ),
        merge_mode="concat",
    )(input_layer)


def create_timed_layer(
    name: str, input_layer, target_layer=Flatten()
) -> TimeDistributed:
    return TimeDistributed(layer=target_layer, name=name)(input_layer)


def create_dense_layer(
    name: str, input_layer, output_size=32, kernel_initializer=CONV_KERNEL_INIT
) -> Dense:
    return Dense(units=output_size, kernel_initializer=kernel_initializer, name=name)(
        input_layer
    )


def ctc_lambda_func(args):
    y_pred, labels, input_length, label_length = args
    y_pred = y_pred[:, :, :]

    return k.ctc_batch_cost(labels, y_pred, input_length, label_length)


def ctc(name: str, args) -> Lambda:
    return Lambda(ctc_lambda_func, output_shape=(1,), name=name)(args)


def create_ctc_layer(
    name: str, y_pred, input_labels, input_length, label_length
) -> Lambda:
    return ctc(name, [y_pred, input_labels, input_length, label_length])


def create_concatenate_layer(
    input_layers: list,
    name: str,
    axis: int = -1,
):
    return Concatenate(name=name, axis=axis)(input_layers)
