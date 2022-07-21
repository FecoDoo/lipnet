import core.models.layers as layers
from tensorflow.keras import backend as k
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from core.utils.config import LipNetConfig
from core.utils.types import Path

ADAM_LEARN_RATE = 1e-4
ADAM_F_MOMENTUM = 0.9
ADAM_S_MOMENTUM = 0.999
ADAM_STABILITY = 1e-08


class LipNet(object):
    def __init__(
        self, config: LipNetConfig,
    ):
        input_shape = self.get_input_shape(
            config.frame_count,
            config.image_height,
            config.image_width,
            config.image_channels,
        )
        self.input_layer = layers.create_input_layer("lipnet_input", input_shape)

        self.zero_1 = layers.create_zero_layer(
            name="lipnet_zero_1", input_layer=self.input_layer
        )
        self.conv_1 = layers.create_conv_layer(
            name="lipnet_conv_1", input_layer=self.zero_1, filters=32
        )
        self.batc_1 = layers.create_batc_layer(
            name="lipnet_batc_1", input_layer=self.conv_1
        )
        self.actv_1 = layers.create_actv_layer(
            name="lipnet_actv_1", input_layer=self.batc_1
        )
        self.pool_1 = layers.create_pool_layer(
            name="lipnet_pool_1", input_layer=self.actv_1
        )
        self.drop_1 = layers.create_spatial_drop_layer(
            name="lipnet_drop_1", input_layer=self.pool_1
        )

        self.zero_2 = layers.create_zero_layer(
            name="lipnet_zero_2", input_layer=self.drop_1
        )
        self.conv_2 = layers.create_conv_layer(
            name="lipnet_conv_2", input_layer=self.zero_2, filters=64
        )
        self.batc_2 = layers.create_batc_layer(
            name="lipnet_batc_2", input_layer=self.conv_2
        )
        self.actv_2 = layers.create_actv_layer(
            name="lipnet_actv_2", input_layer=self.batc_2
        )
        self.pool_2 = layers.create_pool_layer(
            name="lipnet_pool_2", input_layer=self.actv_2
        )
        self.drop_2 = layers.create_spatial_drop_layer(
            name="lipnet_drop_2", input_layer=self.pool_2
        )

        self.zero_3 = layers.create_zero_layer(
            name="lipnet_zero_3", input_layer=self.drop_2, padding=(1, 1, 1)
        )
        self.conv_3 = layers.create_conv_layer(
            name="lipnet_conv_3",
            input_layer=self.zero_3,
            filters=96,
            kernel_size=(3, 3, 3),
        )
        self.batc_3 = layers.create_batc_layer(
            name="lipnet_batc_3", input_layer=self.conv_3
        )
        self.actv_3 = layers.create_actv_layer(
            name="lipnet_actv_3", input_layer=self.batc_3
        )
        self.pool_3 = layers.create_pool_layer(
            name="lipnet_pool_3", input_layer=self.actv_3
        )
        self.drop_3 = layers.create_spatial_drop_layer(
            name="lipnet_drop_3", input_layer=self.pool_3
        )

        self.timed_0 = layers.create_timed_layer(
            name="lipnet_timed_0", input_layer=self.drop_3
        )

        self.gru_1 = layers.create_bi_gru_layer(
            name="lipnet_gru_1", input_layer=self.timed_0
        )
        self.gru_1_actv = layers.create_actv_layer(
            name="lipnet_gru_1_actv", input_layer=self.gru_1
        )
        self.gru_2 = layers.create_bi_gru_layer(
            name="lipnet_gru_2", input_layer=self.gru_1_actv
        )
        self.gru_2_actv = layers.create_actv_layer(
            name="lipnet_gru_2_actv", input_layer=self.gru_2
        )

        self.dense_1 = layers.create_dense_layer(
            name="lipnet_dense_1",
            input_layer=self.gru_2_actv,
            output_size=config.output_size,
        )
        self.y_pred = layers.create_actv_layer(
            name="lipnet_softmax", input_layer=self.dense_1, activation="softmax"
        )

        self.input_labels = layers.create_input_layer(
            name="lipnet_labels", shape=[config.max_string], dtype="uint8"
        )
        self.input_length = layers.create_input_layer(
            name="lipnet_input_length", shape=[1], dtype="uint8"
        )
        self.label_length = layers.create_input_layer(
            name="lipnet_label_length", shape=[1], dtype="uint8"
        )

        self.loss_out = layers.create_ctc_layer(
            name="lipnet_ctc",
            y_pred=self.y_pred,
            input_labels=self.input_labels,
            input_length=self.input_length,
            label_length=self.label_length,
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

    def compile(self, optimizer=None, *args, **kwargs):
        if optimizer is None:
            optimizer = Adam(
                learning_rate=ADAM_LEARN_RATE,
                beta_1=ADAM_F_MOMENTUM,
                beta_2=ADAM_S_MOMENTUM,
                epsilon=ADAM_STABILITY,
            )

        self.model.compile(
            loss={"lipnet_ctc": lambda inputs, outputs: outputs},
            optimizer=optimizer,
            metrics=["accuracy"],
            *args,
            **kwargs
        )

    def fit(self, *args, **kwargs):
        self.model.fit(*args, **kwargs)

    def load_weights(self, path: Path, *args, **kwargs):
        if not path.exists():
            raise FileNotFoundError("model weights not found")

        self.model.load_weights(path, *args, **kwargs)

    def summary(self, *args, **kwargs):
        return self.model.summary(*args, **kwargs)

    @staticmethod
    def get_input_shape(
        frame_count: int, image_height: int, image_width: int, image_channels: int
    ) -> tuple:
        if k.image_data_format() == "channels_first":
            return image_channels, frame_count, image_height, image_width
        else:
            return frame_count, image_height, image_width, image_channels

    def predict(self, input_batch):
        return self.get_pre_softmax_layer_output([input_batch])[0]

    @property
    def get_pre_softmax_layer_output(self):
        return k.function(inputs=[self.input_layer], outputs=[self.dense_1])

    @property
    def lipnet_timed_layer(self):
        return self.timed_0

    @property
    def layers(self):
        return self.model.layers

    @property
    def input(self):
        return self.input_layer

    @property
    def output(self):
        return self.y_pred
