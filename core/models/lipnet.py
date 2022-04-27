import os
import core.models.layers as layers
from tensorflow.keras import backend as k
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from core.utils.config import LipNetConfig

ADAM_LEARN_RATE = 0.0001
ADAM_F_MOMENTUM = 0.9
ADAM_S_MOMENTUM = 0.999
ADAM_STABILITY = 1e-08


class LipNet(object):
    def __init__(
        self,
        config: LipNetConfig,
    ):
        input_shape = self.get_input_shape(
            config.frame_count,            
            config.image_width,
            config.image_height,
            config.image_channels,
        )
        self.input_layer = layers.create_input_layer("lipnet_input", input_shape)

        self.zero_1 = layers.create_zero_layer("lipnet_zero_1", self.input_layer)
        self.conv_1 = layers.create_conv_layer("lipnet_conv_1", self.zero_1, 32)
        self.batc_1 = layers.create_batc_layer("lipnet_batc_1", self.conv_1)
        self.actv_1 = layers.create_actv_layer("lipnet_actv_1", self.batc_1)
        self.pool_1 = layers.create_pool_layer("lipnet_pool_1", self.actv_1)
        self.drop_1 = layers.create_spatial_drop_layer("lipnet_drop_1", self.pool_1)

        self.zero_2 = layers.create_zero_layer("lipnet_zero_2", self.drop_1)
        self.conv_2 = layers.create_conv_layer("lipnet_conv_2", self.zero_2, 64)
        self.batc_2 = layers.create_batc_layer("lipnet_batc_2", self.conv_2)
        self.actv_2 = layers.create_actv_layer("lipnet_actv_2", self.batc_2)
        self.pool_2 = layers.create_pool_layer("lipnet_pool_2", self.actv_2)
        self.drop_2 = layers.create_spatial_drop_layer("lipnet_drop_2", self.pool_2)

        self.zero_3 = layers.create_zero_layer(
            "lipnet_zero_3", self.drop_2, padding=(1, 1, 1)
        )
        self.conv_3 = layers.create_conv_layer(
            "conv_3", self.zero_3, 96, kernel_size=(3, 3, 3)
        )
        self.batc_3 = layers.create_batc_layer("lipnet_batc_3", self.conv_3)
        self.actv_3 = layers.create_actv_layer("lipnet_actv_3", self.batc_3)
        self.pool_3 = layers.create_pool_layer("lipnet_pool_3", self.actv_3)
        self.drop_3 = layers.create_spatial_drop_layer("lipnet_drop_3", self.pool_3)

        self.timed_0 = layers.create_timed_layer(self.drop_3)

        self.gru_1 = layers.create_bi_gru_layer("lipnet_gru_1", self.timed_0)
        self.gru_1_actv = layers.create_actv_layer("lipnet_gru_1_actv", self.gru_1)
        self.gru_2 = layers.create_bi_gru_layer("lipnet_gru_2", self.gru_1_actv)
        self.gru_2_actv = layers.create_actv_layer("lipnet_gru_2_actv", self.gru_2)

        self.dense_1 = layers.create_dense_layer(
            "lipnet_dense_1", self.gru_2_actv, config.output_size
        )
        self.y_pred = layers.create_actv_layer(
            "lipnet_softmax", self.dense_1, activation="softmax"
        )

        self.input_labels = layers.create_input_layer(
            "lipnet_labels", shape=[config.max_string]
        )
        self.input_length = layers.create_input_layer(
            "lipnet_input_length", shape=[1], dtype="int64"
        )
        self.label_length = layers.create_input_layer(
            "lipnet_label_length", shape=[1], dtype="int64"
        )

        self.loss_out = layers.create_ctc_layer(
            "lipnet_ctc",
            self.y_pred,
            self.input_labels,
            self.input_length,
            self.label_length,
        )

        self.model = Model(
            inputs=[
                self.input_layer,
                self.input_labels,
                self.input_length,
                self.label_length,
            ],
            outputs=self.loss_out,
        )

        self.output = self.timed_0

    def compile(self, optimizer=None):
        if optimizer is None:
            optimizer = Adam(
                learning_rate=ADAM_LEARN_RATE,
                beta_1=ADAM_F_MOMENTUM,
                beta_2=ADAM_S_MOMENTUM,
                epsilon=ADAM_STABILITY,
            )

        self.model.compile(
            loss={"ctc": lambda inputs, outputs: outputs}, optimizer=optimizer
        )

    def fit(self, *args, **kw):
        self.model.fit(*args, **kw)

    def load_weights(self, path: os.PathLike):
        if not path.exists():
            raise FileNotFoundError("model weights not found")

        self.model.load_weights(path)

    def summary(self):
        return self.model.summary()
    

    @staticmethod
    def get_input_shape(
        frame_count: int, image_width: int, image_height: int, image_channels: int
    ) -> tuple:
        if k.image_data_format() == "channels_first":
            return image_channels, frame_count, image_width, image_height
        else:
            return frame_count, image_width, image_height, image_channels

    def predict(self, input_batch):
        return self.capture_softmax_output([input_batch])[0]

    @property
    def capture_softmax_output(self):
        return k.function(inputs=[self.input_layer], outputs=[self.y_pred])
