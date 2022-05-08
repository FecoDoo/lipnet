import os
from numpy import ndarray
from tensorflow.keras.applications.densenet import DenseNet121, preprocess_input
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from tensorflow.keras.backend import function, image_data_format
from tensorflow.keras.layers import Dense, Dropout
from core.utils.types import Stream

# params
ADAM_LEARN_RATE = 1e-3

class DenseNet(object):
    def __init__(self, config) -> None:
        input_shape = self.get_input_shape(          
            config.image_height,
            config.image_width,
            config.image_channels,
        )

        self.base_model = DenseNet121(
            weights="imagenet",
            include_top=False,
            input_shape=input_shape,
            pooling="avg",
        )

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

        self.densenet_custom_output = Dense(
            name="densenet_custom_output",
            units=7,
            activation="softmax",
        )(self.densenet_custom_dropout_2)

        self.model = Model(inputs=self.base_model.input, outputs=self.densenet_custom_output)

    def compile(self):
        self.model.compile(
            loss=categorical_crossentropy, optimizer=Adam(learning_rate=ADAM_LEARN_RATE)
        )
    
    def fit(
        self,
        *args,
        **kwargs
    ):
        return self.model.fit(*args,**kwargs)
        
    def summary(self):
        return self.model.summary()

    def load_weights(self, path: os.PathLike):
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
    def input(self):
        return self.model.input

    @property
    def output(self):
        return self.model.output

    @property
    def layers(self):
        return self.model.layers

    @property
    def pre_softmax_layer(self):
        return self.densenet_custom_dropout_2
    
    @property
    def pre_softmax_layer_output(self):
        return function(
            inputs=[self.input], outputs=[self.densenet_custom_dropout_2]
        )

    def get_pre_softmax_layer_output(self, input_batch):
        return self.pre_softmax_layer_output([input_batch])[0]


    def predict(self, input_batch: Stream):

        if isinstance(input_batch, ndarray):
            input_batch = preprocess_input(input_batch)

        return self.model.predict(input_batch)
