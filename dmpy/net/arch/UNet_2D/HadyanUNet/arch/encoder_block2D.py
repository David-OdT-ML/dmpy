from tensorflow.keras.layers import Conv2D, MaxPool2D


def encoder_block2D(x, filters, kernel_size=(3, 3), padding="same", strides=1, activationFn="relu"):
    c = Conv2D(filters, kernel_size, padding=padding, strides=strides, activation=activationFn)(x)
    c = Conv2D(filters, kernel_size, padding=padding, strides=strides, activation=activationFn)(c)
    p = MaxPool2D((2, 2), (2, 2))(c)
    return c, p