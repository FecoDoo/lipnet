from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.backend import image_data_format
from core.utils.config import BaselineConfig, LipNetConfig
from core.utils.types import Stream, Union, Path
from core.models.lipnet import LipNet
from tensorflow.keras.layers import TimeDistributed, Dropout, Dense, GRU, Bidirectional
from tensorflow.keras.layers import Input, Concatenate


class DNN(object):
    def __init__(
        self,
        lipnet_config: LipNetConfig,
        baseline_config: BaselineConfig,
        lipnet_model_weight: Path,
        baseline_model_weight: Path,
    ) -> None:

        baseline_input_shape = self.get_input_shape(
            lipnet_config.frame_count,
            baseline_config.image_height,
            baseline_config.image_width,
            baseline_config.image_channels,
        )

        # load model
        baseline = load_model(baseline_model_weight)

        self.baseline = Model(baseline.input, baseline.get_layer("feats").output)

        self.lipnet = LipNet(lipnet_config)
        self.lipnet.load_weights(lipnet_model_weight)

        self.fix_internal_model_layers(self.baseline)
        self.fix_internal_model_layers(self.lipnet)

        ############################################
        self.dnn_baseline_input_layer = Input(
            name="dnn_baseline_input_layer", shape=baseline_input_shape
        )

        self.dnn_baseline_time_distributed_layer = TimeDistributed(
            layer=self.baseline, name="dnn_baseline_time_distributed_layer",
        )(self.dnn_baseline_input_layer)

        self.dnn_baseline_bidirectional_layer = Bidirectional(
            GRU(
                units=256,
                return_sequences=True,
                activation="tanh",
                kernel_initializer="Orthogonal",
                name="dnn_baseline_gru_layer",
            ),
            merge_mode="concat",
            name="dnn_baseline_bidirectional_layer",
        )(self.dnn_baseline_time_distributed_layer)

        ############################################
        # combining outputs from both models
        self.dnn_concatenate_layer = Concatenate(name="dnn_concatenate_layer", axis=2,)(
            [self.lipnet.lipnet_timed_layer, self.dnn_baseline_bidirectional_layer]
        )

        self.dnn_gru_layer = Bidirectional(
            GRU(
                units=512,
                return_sequences=False,
                activation="tanh",
                kernel_initializer="Orthogonal",
                name="dnn_gru_layer",
            ),
            merge_mode="concat",
            name="dnn_bidirectional_layer",
        )(self.dnn_concatenate_layer)

        ##########################################

        self.dnn_dense_0 = Dense(
            name="dnn_dense_0", units=512, kernel_initializer="he_normal"
        )(self.dnn_gru_layer)

        # add fc layers
        self.dnn_dropout_0 = Dropout(name="dnn_dropout_0", rate=0.5,)(self.dnn_dense_0)

        self.dnn_dense_1 = Dense(
            name="dnn_dense_1", units=256, kernel_initializer="he_normal"
        )(self.dnn_dropout_0)

        self.dnn_output = Dense(name="dnn_output", units=7, activation="softmax",)(
            self.dnn_dense_1
        )

        self.model = Model(
            inputs=[self.dnn_baseline_input_layer, self.lipnet.input],
            outputs=[self.dnn_output],
        )

    def fix_internal_model_layers(self, model):
        for layer in model.layers:
            layer.trainable = False

    def compile(self, learning_rate: float = 1e-3, *args, **kwargs):
        self.model.compile(
            loss=categorical_crossentropy,
            optimizer=Adam(learning_rate=learning_rate),
            *args,
            **kwargs
        )

    def summary(self, *args, **kwargs):
        return self.model.summary(*args, **kwargs)

    def fit(self, *args, **kwargs):
        return self.model.fit(*args, **kwargs)

    def predict(self, input_batch: Stream, *args, **kwargs):
        return self.model.predict(input_batch, *args, **kwargs)

    @property
    def input(self):
        return self.model.input

    @property
    def output(self):
        return self.model.output

    @property
    def layers(self):
        return self.model.layers

    @staticmethod
    def get_input_shape(
        frame_count: int, image_height: int, image_width: int, image_channels: int
    ) -> tuple:
        if image_data_format() == "channels_first":
            return image_channels, frame_count, image_height, image_width
        else:
            return frame_count, image_height, image_width, image_channels
