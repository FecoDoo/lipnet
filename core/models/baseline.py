import os
from numpy import isin, ndarray
from tensorflow.keras.applications.densenet import DenseNet121, preprocess_input
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from tensorflow.keras.backend import function, image_data_format
from core.utils.types import Stream
# import core.models.layers as layers
from tensorflow.keras.layers import Dense, Dropout

# params
ADAM_LEARN_RATE = 1e-4


class DenseNet(object):
    def __init__(self) -> None:
        # input_shape = self.get_input_shape(
        #     config.frame_count,            
        #     config.image_width,
        #     config.image_height,
        #     config.image_channels,
        # )

        self.base_model = DenseNet121(
            weights="imagenet",
            include_top=False,
            input_shape=(224, 224, 3),
            pooling="avg",
        )

        # self.densenet_custom_dense_0 = layers.create_dense_layer(
        #     name="densenet_custom_dense_0",
        #     input_layer=self.base_model.output,
        #     output_size=256,
        #     activation="relu",
        # )
        # self.densenet_custom_dropout_0 = layers.create_drop_layer(
        #     name="densenet_custom_dropout_0",
        #     input_layer=self.densenet_custom_dense_0,
        #     drop_rate=0.7,
        # )

        # self.densenet_custom_dense_1 = layers.create_dense_layer(
        #     name="densenet_custom_dense_1",
        #     input_layer=self.densenet_custom_dropout_0,
        #     output_size=128,
        #     activation="relu",
        # )
        # self.densenet_custom_dropout_1 = layers.create_drop_layer(
        #     name="densenet_custom_dropout_1",
        #     input_layer=self.densenet_custom_dense_1,
        #     drop_rate=0.5,
        # )

        # self.densenet_custom_dense_2 = layers.create_dense_layer(
        #     name="densenet_custom_dense_2",
        #     input_layer=self.densenet_custom_dropout_1,
        #     output_size=64,
        #     activation="relu",
        # )
        # self.densenet_custom_dropout_2 = layers.create_drop_layer(
        #     name="densenet_custom_dropout_2",
        #     input_layer=self.densenet_custom_dense_2,
        #     drop_rate=0.3,
        # )

        # self.y_pred = layers.create_dense_layer(
        #     name="densenet_custom_y_pred",
        #     input_layer=self.densenet_custom_dropout_2,
        #     output_size=7,
        #     activation="softmax",
        # )

        self.densenet_custom_dense_0 = Dense(
            name="densenet_custom_dense_0",
            units=256,
            activation="relu",
        )(self.base_model.output)

        self.densenet_custom_dropout_0 = Dropout(
            name="densenet_custom_dropout_0",
            rate=0.7,
        )(self.densenet_custom_dense_0)

        self.densenet_custom_dense_1 = Dense(
            name="densenet_custom_dense_1",
            units=128,
            activation="relu",
        )(self.densenet_custom_dropout_0)

        self.densenet_custom_dropout_1 = Dropout(
            name="densenet_custom_dropout_1",
            rate=0.5,
        )(self.densenet_custom_dense_1)

        self.densenet_custom_dense_2 = Dense(
            name="densenet_custom_dense_2",
            units=64,
            activation="relu",
        )(self.densenet_custom_dropout_1)

        self.densenet_custom_dropout_2 = Dropout(
            name="densenet_custom_dropout_2",
            rate=0.3,
        )(self.densenet_custom_dense_2)

        self.y_pred = Dense(
            name="densenet_custom_y_pred",
            units=7,
            activation="softmax",
        )(self.densenet_custom_dropout_2)

        self.model = Model(inputs=self.base_model.input, outputs=self.y_pred)

        self.output = self.densenet_custom_dropout_2

    def compile(self):
        self.model.compile(
            loss=categorical_crossentropy, optimizer=Adam(learning_rate=ADAM_LEARN_RATE)
        )
    
    def summary(self):
        return self.model.summary()

    def load_weights(self, path: os.PathLike):
        if not path.exists():
            raise FileNotFoundError("model weights not found")
            
        self.model.load_weights(path)

    @staticmethod
    def get_input_shape(
        frame_count: int, image_width: int, image_height: int, image_channels: int
    ) -> tuple:
        if image_data_format() == "channels_first":
            return image_channels, frame_count, image_width, image_height
        else:
            return frame_count, image_width, image_height, image_channels

    @property
    def feature(self):
        return function(
            inputs=[self.base_model.input], outputs=[self.densenet_custom_dropout_2]
        )

    def predict(self, input_batch: Stream):

        if isinstance(input_batch, ndarray):
            input_batch = preprocess_input(input_batch)

        return self.feature([input_batch])[0]
