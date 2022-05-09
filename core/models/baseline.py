import os
from dataclasses import dataclass
from numpy import ndarray
from tensorflow.keras import Sequential
from tensorflow.keras.applications.densenet import DenseNet121, preprocess_input
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from tensorflow.keras.backend import function, image_data_format
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Input
from tensorflow.keras.layers import RandomFlip, RandomRotation, RandomContrast, Rescaling
from core.utils.types import Stream

class DenseNet(object):
    def __init__(self, config: dataclass, mode="training") -> None:
        """init baseline model

        Args:
            config (dataclass): pre-trained model config
            mode (str, optional): choose structure for different purpose. Defaults to "training".

        Raises:
            ValueError: if mode is neither training nor predicting
        """
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

        # Freeze the base_model
        self.base_model.trainable = False

        self.densenet_inputs = Input(shape=input_shape, name="densenet_custom_input_layer")
        
        if mode == "training":
            self.densenet_argumentation = Sequential(
                layers=[RandomFlip("horizontal"), RandomRotation(0.1), RandomContrast(0.1)],
                name="densenet_custom_argumentation_layer"
            )(self.densenet_inputs)

            self.densenet_rescaling_layer = Rescaling(
                scale=1 / 127.5,
                offset=-1,
                name="densenet_custom_rescaling_layer"
            )(self.densenet_argumentation)

        elif mode == "predicting":
            self.densenet_rescaling_layer = Rescaling(
                scale=1 / 127.5,
                offset=-1,
                name="densenet_custom_rescaling_layer"
            )(self.densenet_inputs)
        else:
            raise ValueError("mode can only be either training or predicting")
            
        self.densenet_basemodel_output = self.base_model(self.densenet_rescaling_layer, training=False)

        self.densenet_custom_layer_0 = Dense(
            name="densenet_custom_layer_0",
            units=1024,
            activation="relu",
        )(self.densenet_basemodel_output)

        self.densenet_custom_layer_1 = Dropout(
            name="densenet_custom_layer_1",
            rate=0.5,
        )(self.densenet_custom_layer_0)

        self.densenet_custom_layer_2 = Dense(
            name="densenet_custom_layer_2",
            units=512,
            activation="relu",
        )(self.densenet_custom_layer_1)

        self.densenet_custom_output = Dense(
            name="densenet_custom_output",
            units=7,
            activation="softmax",
        )(self.densenet_custom_layer_2)

        self.model = Model(inputs=self.densenet_inputs, outputs=self.densenet_custom_output)

    def compile(self, adam_learning_rate=1e-3):
        self.model.compile(
            loss=categorical_crossentropy,
            optimizer=Adam(learning_rate=adam_learning_rate),
            metrics=["accuracy"]
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
    def trainable(self):
        return self.model.trainable

    @property
    def basemodel(self):
        return self.base_model
    
    @property
    def pre_softmax_layer(self):
        return self.densenet_custom_layer_2
    
    @property
    def pre_softmax_layer_output(self):
        return function(
            inputs=[self.input], outputs=[self.densenet_custom_layer_2]
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