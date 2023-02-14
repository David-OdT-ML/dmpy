from tensorflow.keras.layers import Conv2D, UpSampling2D, Concatenate


def decoder_block2D(x, skip, filters, kernel_size=(3, 3), padding="same", strides=1, activationFn="relu"):
    us = UpSampling2D((2, 2))(x)
    concat = Concatenate()([us, skip])
    c = Conv2D(filters, kernel_size, padding=padding, strides=strides, activation=activationFn)(concat)
    c = Conv2D(filters, kernel_size, padding=padding, strides=strides, activation=activationFn)(c)
    return c