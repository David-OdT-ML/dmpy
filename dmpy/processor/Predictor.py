import numpy as np
from patchify import patchify, unpatchify


class Predictor(object):
    def __init__(self) -> None:
        pass

    def __append_patch(self, model, patches, predicted_patches, i, j):
        single_patch = patches[i, j, :, :]
        single_patch_3ch = np.stack((single_patch,)*3, axis=-1)
        single_patch_3ch_input = np.expand_dims(single_patch_3ch, axis=0)
        single_patch_prediction = model.predict(single_patch_3ch_input)
        single_patch_prediction_argmax = np.argmax(
            single_patch_prediction, axis=3)[0, :, :]
        return predicted_patches.append(single_patch_prediction_argmax)

    def prediction(self, seismic, model, mode='inline'):
        # Append 2D seismic into 3D seismic
        if mode == 'inline':
            predicted = []
            for i in range(seismic.shape[0]):
                val_input = seismic[i, :, :]
                patches = patchify(val_input, (128, 128), step=128)
                predicted_patches = []
                for i in range(patches.shape[0]):
                    for j in range(patches.shape[1]):
                        self.__append_patch(
                            model, patches, predicted_patches, i, j)
                predicted_patches_reshaped = np.reshape(
                    predicted_patches, (patches.shape[0], patches.shape[1], patches.shape[2], patches.shape[3]))
                reconstructed_seismic = unpatchify(
                    predicted_patches_reshaped, val_input.shape)
                predicted.append(reconstructed_seismic)
            return np.array(predicted)
        elif mode == 'crossline':
            predicted = []
            for i in range(seismic.shape[1]):
                val_input = seismic[:, i, :]
                patches = patchify(val_input, (128, 128), step=128)
                predicted_patches = []
                for i in range(patches.shape[0]):
                    for j in range(patches.shape[1]):
                        self.__append_patch(
                            model, patches, predicted_patches, i, j)
                predicted_patches_reshaped = np.reshape(
                    predicted_patches, (patches.shape[0], patches.shape[1], patches.shape[2], patches.shape[3]))
                reconstructed_seismic = unpatchify(
                    predicted_patches_reshaped, val_input.shape)
                predicted.append(reconstructed_seismic)
            return np.array(predicted)

    def merge_2_volumes_simple_binary(self, v1, v2):
        print(v1.shape)
        print(v2.shape)
        v2_transposed = np.moveaxis(v1, 1, 0)
        print(v2.shape)
        combined_volume = v1 + v2_transposed
        combined_volume[combined_volume > 0] = 1
        return combined_volume
