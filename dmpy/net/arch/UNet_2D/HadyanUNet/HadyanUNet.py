import AbstractUNet2D

import arch.encoder_block2D as encoder2D
import arch.decoder_block2D as decoder2D
import arch.bridge_block2D as bridge2D
import arch.hadyan_UNet

class HadyanUNet(AbstractUNet2D):

    def __init__(self,  filters=[16, 32, 64, 128, 256], 
                        kernel_size=(3,3), 
                        strides=(1,1), 
                        output_channels=1, 
                        padding="same", 
                        width=None, 
                        height=None, 
                        depth=1, 
                        input_channels=1, 
                        conv_layers=2, 
                        activationFn="relu", 
                        finalActivationFn="sigmoid"):
        super().__init__(filters, kernel_size, strides, output_channels, padding, width, height, depth, input_channels, conv_layers, activationFn, finalActivationFn)

    def encoder(self, p, filters, kernel_size, padding, strides, conv_layers, activationFn):
        return encoder2D(p, filters, kernel_size, padding, strides, activationFn)
        
    def decoder(self, u, c, filters, kernel_size, padding, strides, conv_layers, activationFn):
        return decoder2D(u, c, filters, kernel_size, padding, strides, conv_layers, activationFn)

    def bridge(x, filters, kernel_size, padding, strides, activationFn):
        return bridge2D(x, filters, kernel_size, padding, strides, activationFn)

    def getModel(self):
        return hadyan_UNet( filters=self.filters, 
                            output_channels=self.output_channels,
                            input_channels=self.input_channels,
                            conv_layers=self.conv_layers,
                            activationFn=self.activationFn)

