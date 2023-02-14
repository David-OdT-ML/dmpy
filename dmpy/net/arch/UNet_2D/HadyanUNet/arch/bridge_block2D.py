from tensorflow.keras.layers import Conv2D


def bridge_block2D(x, filters, kernel_size=(3, 3), padding="same", strides=1, activationFn="relu"):
    c = Conv2D(filters, kernel_size, padding=padding, strides=strides, activation=activationFn)(x)
    c = Conv2D(filters, kernel_size, padding=padding, strides=strides, activation=activationFn)(c)
    return c