from abc import ABC, abstractmethod
from tensorflow.keras.layers import Input, Conv2D
from tensorflow.keras.models import Model

class AbstractUNet2D(ABC):
    def __init__(self, filters=[16, 32, 64, 128, 256], kernel_size=(3,3), strides=(1,1), output_channels=1, padding="same", width=None, height=None, depth=1, input_channels=1, conv_layers=2, activationFn="relu", finalActivationFn="sigmoid"):
        self.name = "UNet 2D"
        self.number_dimensions=2
        self.filters = filters
        self.kernel_size=kernel_size
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.conv_layers = conv_layers
        self.padding="same"
        self.strides=strides
        self.activationFn=activationFn
        self.finalActivationFn=finalActivationFn
        self.shape=(width, height, input_channels)
        self.image_size=(width, height, depth)


    @abstractmethod
    def encoder(self, p, filter, kernel_size, padding, strides, conv_layers, activationFn):
        pass


    @abstractmethod
    def bridge(self, x, filter, kernel_size, padding, strides, activationFn):
        pass


    @abstractmethod
    def decoder(self, u, c, filter, kernel_size, padding, strides, conv_layers, activationFn):
        pass

    @abstractmethod
    def getModel(self):
        return getDefaultModel()

    def getDefaultModel(self):

        inputs = Input(shape=self.shape)

        # encoder
        (c1, p1) = self.encoder(inputs, self.filters[0], conv_layers=self.conv_layers, activation=self.activationFn)
        (c2, p2) = self.encoder(p1,     self.filters[1], conv_layers=self.conv_layers, activation=self.activationFn)
        (c3, p3) = self.encoder(p2,     self.filters[2], conv_layers=self.conv_layers, activation=self.activationFn)
        (c4, p4) = self.encoder(p3,     self.filters[3], conv_layers=self.conv_layers, activation=self.activationFn)

        # bridge
        br5 = self.encoder(p4, self.filters[4], conv_layers=self.conv_layers, activation=self.activationFn)

        # decoder
        c6 = self.decoder(br5, c4, self.filters[3], conv_layers=self.conv_layers, activation=self.activationFn)
        c7 = self.decoder(c6, c3,  self.filters[2], conv_layers=self.conv_layers, activation=self.activationFn)
        c8 = self.decoder(c7, c2,  self.filters[1], conv_layers=self.conv_layers, activation=self.activationFn)
        c9 = self.decoder(c8, c1,  self.filters[0], conv_layers=self.conv_layers, activation=self.activationFn)

        outputs = Conv2D(output_channels=self.output_channels, kernel_size=(1, 1), strides=(1, 1), activation=self.finalActivationFn)(c9)

        model = Model(inputs=inputs, outputs=outputs)

        return model
