from dataclasses import dataclass
from numpy import ndarray
from tensorflow.keras.applications.xception import Xception, preprocess_input
from tensorflow.keras.applications.densenet import DenseNet121, preprocess_input
from tensorflow.keras.applications.vgg19 import VGG19, preprocess_input
from tensorflow.keras.applications.mobilenet import MobileNet, preprocess_input
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from tensorflow.keras.backend import function, image_data_format
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Input
from core.utils.types import Stream, Path
from utils.logger import get_logger


class BaseLineModel(object):
    def __init__(self, config, base_model=None) -> None:
        """init baseline model

        Args:
            config (dataclass): pre-trained model config

        Raises:
            ValueError: if mode is neither training nor predicting
        """
        input_shape = self.get_input_shape(
            config.image_height, config.image_width, config.image_channels,
        )

        self.logger = get_logger(name="baseline")

        self.base_model = base_model

        # Freeze the base_model
        self.base_model.trainable = False

        self.baseline_custom_input_lauyer = Input(
            shape=input_shape, name="baseline_custom_input_lauyer"
        )

        self.baseline_basemodel_output = self.base_model(
            self.baseline_custom_input_lauyer, training=False
        )

        self.baseline_custom_dense_layer_0 = Dense(
            name="baseline_custom_dense_layer_0", units=4096, activation="relu",
        )(self.baseline_basemodel_output)

        self.baseline_custom_dense_layer_1 = Dense(
            name="baseline_custom_dense_layer_1", units=2048, activation="relu",
        )(self.baseline_custom_dense_layer_0)

        self.baseline_custom_dropout_layer_0 = Dropout(
            name="baseline_custom_dropout_layer_0", rate=0.5,
        )(self.baseline_custom_dense_layer_1)

        self.baseline_custom_batchnorm_layer_0 = BatchNormalization(
            name="baseline_custom_batchnorm_layer_0",
        )(self.baseline_custom_dropout_layer_0)

        self.baseline_custom_output_layer = Dense(
            name="baseline_custom_output_layer", units=7, activation="softmax",
        )(self.baseline_custom_batchnorm_layer_0)

        self.model = Model(
            inputs=self.baseline_custom_input_lauyer,
            outputs=self.baseline_custom_output_layer,
        )

    def compile(self, adam_learning_rate=1e-3):
        self.model.compile(
            loss=categorical_crossentropy,
            optimizer=Adam(learning_rate=adam_learning_rate),
            metrics=["accuracy"],
        )

    def fit(self, *args, **kwargs):
        return self.model.fit(*args, **kwargs)

    def summary(self):
        return self.model.summary()

    def load_weights(self, path: Path):
        if not path.exists():
            raise FileNotFoundError("model weights not found")

        self.model.load_weights(path)

    @staticmethod
    def get_input_shape(
        image_height: int, image_width: int, image_channels: int
    ) -> tuple:
        if image_data_format() == "channels_first":
            return image_channels, image_height, image_width
        else:
            return image_height, image_width, image_channels

    @property
    def basemodel_name(self):
        return self.base_model.name

    @property
    def input(self):
        return self.model.input

    @property
    def output(self):
        return self.model.output

    @property
    def layers(self):
        return self.model.layers

    @property
    def trainable(self):
        return self.model.trainable

    @property
    def basemodel(self):
        return self.base_model

    @property
    def pre_softmax_layer(self):
        return self.baseline_custom_batchnorm_layer_0

    @property
    def pre_softmax_layer_output(self):
        return function(
            inputs=[self.input], outputs=[self.baseline_custom_batchnorm_layer_0]
        )

    def get_pre_softmax_layer_output(self, input_batch):
        return self.pre_softmax_layer_output([input_batch])[0]

    def predict(self, input_batch: Stream):

        if isinstance(input_batch, ndarray):
            input_batch = preprocess_input(input_batch)

        return self.model.predict(input_batch)

    def evaluate(self, input_batch: Stream):

        if isinstance(input_batch, ndarray):
            input_batch = preprocess_input(input_batch)

        return self.model.evaluate(input_batch)


class DenseNet_Baseline(BaseLineModel):
    def __init__(self, config: dataclass) -> None:

        base_model = DenseNet121(weights="imagenet", include_top=False, pooling="avg",)

        super().__init__(config, base_model)


class Xception_Baseline(BaseLineModel):
    def __init__(self, config: dataclass) -> None:

        base_model = Xception(weights="imagenet", include_top=False, pooling="avg",)

        super().__init__(config, base_model)


class VGG19_Baseline(BaseLineModel):
    def __init__(self, config: dataclass) -> None:

        base_model = VGG19(weights="imagenet", include_top=False, pooling="avg",)

        super().__init__(config, base_model)


class MobileNet_Baseline(BaseLineModel):
    def __init__(self, config: dataclass) -> None:

        base_model = MobileNet(weights="imagenet", include_top=False, pooling="avg",)

        super().__init__(config, base_model)
