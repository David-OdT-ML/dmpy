from dmpy.dmpy.net.variant.UNet_HadyanUnet2D.block.encoder_block2D import encoder2D
from tensorflow.keras.layers import Input, Conv2D
from tensorflow.keras.models import Model

import encoder_block2D as encoder
import decoder_block2D as decoder
import bridge_block2D as bridge

def hadyan_UNetArch(filters=[16, 32, 64, 128, 256], output_channels=1, width=None, height=None, input_channels=1, conv_layers=2, activationFn="relu"):
    
    inputs = Input(shape=(width, height, input_channels))
    
    # encoder
    p0 = inputs
    c1, p1 = encoder(p0, filters[0], activationFn=activationFn) #128 -> 64
    c2, p2 = encoder(p1, filters[1], activationFn=activationFn) #64 -> 32
    c3, p3 = encoder(p2, filters[2], activationFn=activationFn) #32 -> 16
    c4, p4 = encoder(p3, filters[3], activationFn=activationFn) #16->8
    
    # bridge
    bn = bridge(p4, filters[4])
    
    # decoder
    u1 = decoder(bn, c4, filters[3], activationFn=activationFn) #8 -> 16
    u2 = decoder(u1, c3, filters[2], activationFn=activationFn) #16 -> 32
    u3 = decoder(u2, c2, filters[1], activationFn=activationFn) #32 -> 64
    u4 = decoder(u3, c1, filters[0], activationFn=activationFn) #64 -> 128
    
    outputs = Conv2D(output_channels, kernel_size=(1, 1), strides=(1, 1), activation="sigmoid")(u4)
    
    model = Model(inputs, outputs)

    return model