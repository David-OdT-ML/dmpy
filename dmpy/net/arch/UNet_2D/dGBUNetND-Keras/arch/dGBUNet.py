from keras.models import Model
from keras.layers import *
import dgbpy.dgbkeras as dgbkeras
from dgbpy.keras_classes import DataPredType
from dmpyseismic.net.abstract.AbstractMonolithic import AbstractMonolithic


class dGBUNet(AbstractMonolithic):

    def build(model_shape, nroutputs, predtype, modelsize=None):

        unet_smallsz = (2, 64)
        unet_mediumsz = (16, 512)
        unet_largesz = (32, 512)

        input_shape = model_shape
        data_format = 'channels_last'

        ndim = dgbkeras.getModelDims(model_shape, data_format)

        conv = pool = upsamp = None
        if ndim == 3:
            conv = Conv3D
            pool = MaxPooling3D
            upsamp = UpSampling3D
        elif ndim == 2:
            conv = Conv2D
            pool = MaxPooling2D
            upsamp = UpSampling2D
        elif ndim == 1:
            conv = Conv1D
            pool = MaxPooling1D
            upsamp = UpSampling1D

        if modelsize == None:
            unetnszs = unet_mediumsz
        else:
            unetnszs = modelsize

        poolsz1 = 2
        poolsz2 = 2
        poolsz3 = 2
        upscalesz = 2
        filtersz1 = unetnszs[0]
        filtersz2 = filtersz1 * poolsz2
        filtersz3 = filtersz2 * poolsz3
        filtersz4 = unetnszs[1]
        axis = -1

        params = dict(kernel_size=3, activation='relu',
                      padding='same', data_format=data_format)

        inputs = Input(input_shape)
        conv1 = conv(filtersz1, **params)(inputs)
        conv1 = conv(filtersz1, **params)(conv1)

        pool1 = pool(pool_size=poolsz1, data_format=data_format)(conv1)

        conv2 = conv(filtersz2, **params)(pool1)
        conv2 = conv(filtersz2, **params)(conv2)
        pool2 = pool(pool_size=poolsz2, data_format=data_format)(conv2)

        conv3 = conv(filtersz3, **params)(pool2)
        conv3 = conv(filtersz3, **params)(conv3)
        pool3 = pool(pool_size=poolsz3, data_format=data_format)(conv3)

        conv4 = conv(filtersz4, **params)(pool3)
        conv4 = conv(filtersz4, **params)(conv4)

        up5 = concatenate(
            [upsamp(size=upscalesz, data_format=data_format)(conv4), conv3], axis=axis)
        conv5 = conv(filtersz3, **params)(up5)
        conv5 = conv(filtersz3, **params)(conv5)

        up6 = concatenate(
            [upsamp(size=poolsz2, data_format=data_format)(conv5), conv2], axis=axis)
        conv6 = conv(filtersz2, **params)(up6)
        conv6 = conv(filtersz2, **params)(conv6)

        up7 = concatenate(
            [upsamp(size=poolsz1, data_format=data_format)(conv6), conv1], axis=axis)
        conv7 = conv(filtersz1, **params)(up7)
        conv7 = conv(filtersz1, **params)(conv7)

        if isinstance(predtype, DataPredType) and predtype == DataPredType.Continuous:
            nrout = nroutputs
            activation = 'linear'
        else:
            if nroutputs == 2:
                nrout = 1
                activation = 'sigmoid'
            else:
                nrout = nroutputs
                activation = 'softmax'
        conv8 = conv(nrout, 1, activation=activation,
                     data_format=data_format)(conv7)

        model = Model(inputs=[inputs], outputs=[conv8])

        return model
