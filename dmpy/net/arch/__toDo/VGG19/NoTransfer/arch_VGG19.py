def vgg19_model():
    
    img_input = Input((image_size, image_size, 3))
        
    # Block 1
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1')(img_input)
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2')(x)
    x = MaxPool2D((2, 2), strides=(2, 2), name='block1_pool')(x)

    # Block 2
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1')(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2')(x)
    x = MaxPool2D((2, 2), strides=(2, 2), name='block2_pool')(x)

    # Block 3
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv4')(x)
    pool3 = MaxPool2D((2, 2), strides=(2, 2), name='block3_pool')(x)

    # Block 4
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1')(pool3)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv4')(x)
    pool4 = MaxPool2D((2, 2), strides=(2, 2), name='block4_pool')(x)

    # Block 5
    x = Conv2D(1024, (3, 3), activation='relu', padding='same', name='block5_conv1')(pool4)
    x = Conv2D(1024, (3, 3), activation='relu', padding='same', name='block5_conv2')(x)
    x = Conv2D(1024, (3, 3), activation='relu', padding='same', name='block5_conv3')(x)
    x = Conv2D(1024, (3, 3), activation='relu', padding='same', name='block5_conv4')(x)
    pool5 = MaxPool2D((2, 2), strides=(2, 2), name='block5_pool')(x)

    # decoder
    score_c5 = Conv2D(2, (1, 1), strides=(1, 1), padding='same', kernel_initializer='zeros')(pool5)
    up_c5 = Conv2DTranspose(2, (2, 2), strides=(2, 2), padding='valid')(score_c5)
    
    score_c4 = Conv2D(2, (1, 1), strides=(1, 1), padding='same', kernel_initializer='zeros')(pool4)
    fuse_16 = Add()([score_c4, up_c5])
    up_c4 = Conv2DTranspose(2, (2, 2), strides=(2, 2), padding='valid')(fuse_16)
    
    score_c3 = Conv2D(2, (1, 1), strides=(1, 1), padding='same', kernel_initializer='zeros')(pool3)
    fuse_32 = Add()([score_c3, up_c4])
    up_c3 = Conv2DTranspose(2, (8, 8), strides=(8, 8), padding='valid', activation='sigmoid')(fuse_32)

    model = Model(inputs=img_input, outputs=up_c3, name='vgg19')
    
    return model

model_vgg = vgg19_model()
model_vgg.compile(optimizer="adam", loss="binary_crossentropy", metrics=["acc"])
plot_model(model_vgg)
history_2 = model_vgg.fit(X_train_norm, 
                    y_train,
                    batch_size=batch_size, 
                    epochs=100,
                    verbose=1,
                    validation_data=(X_test_norm, y_test), ) # callbacks=EarlyStopping(monitor='val_acc', mode='max',patience=10))