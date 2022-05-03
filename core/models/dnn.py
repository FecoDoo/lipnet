import os

from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from tensorflow.keras.backend import image_data_format

from core.utils.config import BaselineConfig
from core.utils.config import BaselineConfig, LipNetConfig
from core.utils.types import Stream
from core.models.lipnet import LipNet
from core.models.baseline import DenseNet
import core.models.layers as layers


class DNN(object):
    def __init__(
        self,
        lipnet_config: LipNetConfig,
        baseline_config: BaselineConfig,
        *args,
        **kwargs
    ) -> None:
        lipnet_input_shape = self.get_input_shape(
            lipnet_config.frame_count,            
            lipnet_config.image_height,
            lipnet_config.image_width,
            lipnet_config.image_channels
        )

        baseline_input_shape = self.get_input_shape(
            lipnet_config.frame_count,
            baseline_config.image_height,
            baseline_config.image_width,
            baseline_config.image_channels
        )

        self.adam_learning_rate = 1e-3

        # load model
        self.baseline = DenseNet(baseline_config)
        self.lipnet = LipNet(lipnet_config)

        self.fix_internal_model_layers(self.baseline)
        self.fix_internal_model_layers(self.lipnet)


        # generate timedistributed wrapper for baseline input
        self.dnn_baseline_input_layer = layers.create_input_layer(name="dnn_baseline_input_layer", shape=baseline_input_shape)
        self.dnn_baseline_time_distributed_layer = layers.create_timed_layer(
            target_layer=self.baseline.model,
            input_layer=self.dnn_baseline_input_layer,
            name="dnn_baseline_time_distributed_layer",
        )
        self.dnn_baseline_gru_layer = layers.create_bi_gru_layer(name="dnn_baseline_gru_layer", input_layer=self.dnn_baseline_time_distributed_layer)

        # combining outputs from both models
        self.dnn_concatenate_layer = layers.create_concatenate_layer(
            name="dnn_concatenate_layer",
            input_layers=[self.lipnet.lipnet_timed_layer, self.dnn_baseline_gru_layer],
            axis=2,
        )

        # add fc layers
        self.dnn_dropout_0 = layers.create_drop_layer(
            name="dnn_dropout_0",
            input_layer=self.dnn_concatenate_layer,
            rate=0.2,
        )
        self.dnn_dense_0 = layers.create_dense_layer(
            name="dnn_dense_0",
            input_layer=self.dnn_dropout_0,
            output_size=128
        )

        self.dnn_output = layers.create_softmax_layer(
            name="dnn_output",
            input_layer=self.dnn_dense_0,
            units=7,
            activation="softmax",
        )

        self.model = Model(
            inputs=[self.lipnet.input, self.dnn_baseline_input_layer], outputs=[self.dnn_output]
        )
    
    def fix_internal_model_layers(self, model):
        for layer in model.layers:
            layer.trainable = False

    def compile(self):
        self.model.compile(
            loss=categorical_crossentropy, optimizer=Adam(learning_rate=self.adam_learning_rate)
        )
        

    def summary(self):
        return self.model.summary()

    def load_weights(self, lipnet_weight_path: os.PathLike, baseline_weight_path):

        if not lipnet_weight_path.is_file():
            raise FileNotFoundError("lipnet weight file does not exist")

        if not baseline_weight_path.is_file():
            raise FileNotFoundError("baseline weight file does not exist")

        self.lipnet.load_weights(lipnet_weight_path)
        self.baseline.load_weights(baseline_weight_path)
    
    def predict(self, input_batch: Stream):
        return self.model.predict(input_batch)

    def summary(self):
        return self.model.summary()

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
        frame_count: int,
        image_height: int,
        image_width: int,
        image_channels: int
    ) -> tuple:
        if image_data_format() == "channels_first":
            return image_channels, frame_count, image_height, image_width
        else:
            return frame_count, image_height, image_width, image_channels