import numpy as np


# Normalization
def normalization(data, clip_factor=2.5):
    clipFactor = clip_factor
    std0 = np.std(data)
    stdClip = std0 * clipFactor
    clippedCube = np.clip(data, -stdClip, +stdClip)
    return np.asarray(clippedCube / stdClip, dtype='single')


def normalize_data(data, mean, std):
    data -= mean[:, np.newaxis, np.newaxis, np.newaxis]
    data /= std[:, np.newaxis, np.newaxis, np.newaxis]
    return data


def normalize_data_storage(data_storage):
    means = list()
    stds = list()
    for index in range(data_storage.shape[0]):
        data = data_storage[index]
        means.append(data.mean(axis=(1, 2, 3)))
        stds.append(data.std(axis=(1, 2, 3)))
    mean = np.asarray(means).mean(axis=0)
    std = np.asarray(stds).mean(axis=0)
    for index in range(data_storage.shape[0]):
        data_storage[index] = normalize_data(data_storage[index], mean, std)
    return data_storage