def conv_block(input, num_filters):
    x = Conv2D(num_filters, 3, padding="same")(input)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x = Conv2D(num_filters, 3, padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    return x

def decoder_block(input, skip_features, num_filters):
    x = Conv2DTranspose(num_filters, (2, 2), strides=2, padding="same")(input)
    x = Concatenate()([x, skip_features])
    x = conv_block(x, num_filters)
    return x

def build_vgg19_unet(input_shape, freeze=1):
    """ Input """
    inputs = Input(input_shape)

    """ Pre-trained VGG19 Model """
    vgg19 = VGG19(include_top=False, weights="imagenet", input_tensor=inputs)
    
    """ 
        Select Freeze layer 
        VGG 19, consist of 22 trainable layers
        Freeze = True, all layers freeze not fine tune
        Freeze = False, all layers fine tune
        Freeze = 1, 1 last layers freeze
        Freeze = 21, 1 last layers fine tune, else freeze
    """

    if freeze == True:
      for layer in vgg19.layers:
        layer.trainable = False
    elif freeze == False:
      for layer in vgg19.layers[:]:
        layer.trainable = True
    elif freeze > 0:
      for layer in vgg19.layers[:freeze]:
        layer.trainable = False
    elif freeze < 0:
      # n_freeze = -1*freeze
      for layer in vgg19.layers[freeze:]:
        layer.trainable = False
    # else:
    #   for layer in vgg19.layers[:-1*freeze]:
    #     layer.trainable = False
    
    for layer in vgg19.layers:
      print(layer, layer.trainable)

    """ Encoder """
    s1 = vgg19.get_layer("block1_conv2").output         ## (512 x 512)
    s2 = vgg19.get_layer("block2_conv2").output         ## (256 x 256)
    s3 = vgg19.get_layer("block3_conv4").output         ## (128 x 128)
    s4 = vgg19.get_layer("block4_conv4").output         ## (64 x 64)

    """ Bridge """
    b1 = vgg19.get_layer("block5_conv4").output         ## (32 x 32)

    """ Decoder """
    d1 = decoder_block(b1, s4, 512)                     ## (64 x 64)
    d2 = decoder_block(d1, s3, 256)                     ## (128 x 128)
    d3 = decoder_block(d2, s2, 128)                     ## (256 x 256)
    d4 = decoder_block(d3, s1, 64)                      ## (512 x 512)

    """ Output """
    outputs = Conv2D(2, 1, padding="same", activation="sigmoid")(d4)

    model = Model(inputs, outputs, name="VGG19_U-Net")
    return model

input_shape = (128,128, 3)
model_unet_vgg_1 = build_vgg19_unet(input_shape, freeze=True)
model_unet_vgg_1.summary()
model_unet_vgg_1.compile(optimizer="adam", loss="binary_crossentropy", metrics=["acc"])
model_unet_vgg_1.fit(X_train_norm, 
                    y_train,
                    batch_size=8, 
                    epochs=100,
                    verbose=1,
                    validation_data=(X_test_norm, y_test), )