# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np

# class Slice(object):

#     def __init__(self):
#         pass

def extract(incube: np.array, direction: int, slicenum: int, flipFlag:bool = False) -> np.array:
    if direction == 1:      # fast direction
        selection = np.transpose(incube[slicenum, :, :])
    elif direction == 2:    # slow direction
        selection = np.transpose(incube[:, slicenum, :])
    elif direction == 3:    # horizontal slice
        selection = incube[:,:,slicenum]
    else:
        raise Exception('direction should be between 1 and 3. The value was: {}'.format(direction))     
    if flipFlag:
        return np.fliplr(selection)
    else:
        return selection


def plot2D(selection, cmap: str='gray_r'):
    shape = np.shape(selection)
    xlen = (shape[1] / shape[0]) * 5.0
    ylen = 5
    fig = plt.figure(figsize=(xlen, ylen))
    ax = fig.add_subplot(121)
    slice = ax.imshow(selection, cmap)
    fig.colorbar(slice, ax=ax)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.show()


def plotImageTarget(image, target, cmap: str='gray_r'):
    shape = np.shape(image)
    xlen = (shape[1] / shape[0]) * 5.0
    ylen = 5
    fig = plt.figure(figsize=(xlen, ylen))
    plt.rcParams.update({'font.size': 1})
    ax = fig.add_subplot(1, 2, 1)
    ax.imshow(image, cmap)
    #fig.colorbar(slice, ax=ax)
    ax.set_xticks([])
    ax.set_yticks([])
    ax = plt.subplot(1, 2, 2)
    ax.imshow(target, cmap)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.show()    


def plot(incube: np.array, direction: int, slicenum: int, cmap: str='gray_r', flipFlag: bool = False) -> np.array:
    selection = extract(incube, direction, slicenum, flipFlag)
    plot2D(selection, cmap)
    return selection
