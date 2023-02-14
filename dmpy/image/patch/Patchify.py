from typing import Tuple, Union, cast

import numpy as np
from typing import Tuple
from itertools import product
from skimage.util.shape import view_as_windows
import tensorflow as tf

# ideas borrowed from https://github.com/dovahcrow/patchify.py for CPU-based splitting/recovery
# utilize https://www.tensorflow.org/api_docs/python/tf/image for code simplification and keeping as much computation on the GPU as possible
# inspiration from https://www.tensorflow.org/guide/autodiff for GPU-based splitting recovery
# TODO: use these ideas for a pipeline based split-augment-train-all-in-GPU methodology
#       for example, train on odd IL/XL slices with some step, test on even IL/XL slices with some multiple of step (to decimate test)


Imsize = Union[Tuple[int, int], Tuple[int, int, int]]




def extract_patches_np2D(img: np.array, patch_shape: Tuple[int, int] = (3, 3),
                        strides: Tuple[int, int] = (1, 1),
                        rates: Tuple[int, int] = (1, 1)) -> np.array:
    patches = extract_patches_np2D_to_tf(img, patch_shape, strides, rates)
    return np.asarray(patches, dtype=float).squeeze()


def extract_patches_np2D_to_tf(img: np.array, patch_shape: Tuple[int, int] = (3, 3),
                        strides: Tuple[int, int] = (1, 1),
                        rates: Tuple[int, int] = (1, 1)) -> tf.Tensor:        
    shape = np.shape(img)
    img = img.reshape(1, *shape, 1)
    imgtf = tf.convert_to_tensor(img, dtype=float)
    return extract_patches_tf(imgtf, patch_shape, strides, rates)


# return tensor of patches
def extract_patches_tf(img: tf.Tensor,
                        patch_shape: Tuple[int, int] = (3, 3),
                        strides: Tuple[int, int] = (1, 1),
                        rates: Tuple[int, int] = (1, 1)) -> tf.Tensor:
    width, height = patch_shape
    stridex, stridey = strides
    ratex, ratey = rates
    ksizes = [1, width, height, 1]
    strides = [1, stridex, stridey, 1]
    rates = [1, ratex, ratey, 1]
    padding = 'VALID'  # or 'SAME'
    return tf.image.extract_patches(img, ksizes, strides, rates, padding)


def _recover_image_tf(img:tf.Tensor, patches:tf.Tensor, tape:tf.GradientTape,
                        patch_shape: Tuple[int, int] = (3, 3),
                        strides: Tuple[int, int] = (1, 1),
                        rates: Tuple[int, int] = (1, 1)) -> tf.Tensor:
    _img = tf.zeros_like(img)
    _pch = extract_patches_tf(_img, patch_shape, strides, rates)
    grad = tape.gradient(_pch, _img)
    # Divide by grad, to "average" together the overlapping patches
    # otherwise they would simply sum up (dividing by a constant will result 
    # in artifacts)
    return (tape.gradient(_pch, _img, output_gradients=patches) / grad)


def recover_image(img:np.ndarray, patches:np.ndarray, patch_shape: Tuple[int, int] = (3, 3),
                        strides: Tuple[int, int] = (1, 1),
                        rates: Tuple[int, int] = (1, 1)) -> np.ndarray:
    imshape = np.shape(img)
    img = img.reshape(1, *imshape, 1)
    imgtensor = tf.convert_to_tensor(img,  dtype=float)
    patchestensor = tf.convert_to_tensor(patches,  dtype=float)
    with tf.GradientTape(persistent=True) as tape:
            tape.watch(imgtensor)   
            invtensor = _recover_image_tf(imgtensor, patchestensor, tape)
    return np.asarray(invtensor, dtype=float).squeeze()
    

def recover_image_tf(imgtensor:tf.Tensor, patchestensor:tf.Tensor, patch_shape: Tuple[int, int] = (3, 3),
                        strides: Tuple[int, int] = (1, 1),
                        rates: Tuple[int, int] = (1, 1)) -> tf.Tensor:
    with tf.GradientTape(persistent=True) as tape:
            tape.watch(imgtensor) 
            invtensor = _recover_image_tf(imgtensor, patchestensor, tape, patch_shape, strides, rates)
    return invtensor
    

def test_patchify_tf():
    print('\nBegin Test')
    nx = 10
    ny = 11
    print('\nCreate 2D image')
    img = [[[[x * nx + y + 1] for y in range(ny)] for x in range(nx)]]
    print(np.asarray(img, dtype=float).squeeze())
    #img = [[[[x * ny + y + 1] for y in range(ny)] for x in range(nx)]]
    print('\nConnvert 2D image to 4D tensor')
    imgtensor = tf.convert_to_tensor(img,  dtype=float)
    with tf.GradientTape(persistent=True) as tape:
        with tape.stop_recording():
            print('\ntf hack: set tape watch')
            tape.watch(imgtensor)
            print('\nDisplay 2D image')
            print(tf.squeeze(imgtensor))
            print('\nExtract and display patches')
            patches = extract_patches_tf(imgtensor)
            print(patches)
            print('\ntf hack: use monitored gradients to reassemble the image')
            invtensor = _recover_image_tf(imgtensor, patches, tape)
            print('\nDisplay reassembled tensor')
            print(invtensor)
            print('\nDisplay reassembled image')
            inv = tf.squeeze(invtensor).numpy()
            print(inv)
            img = np.asarray(img, dtype=float).squeeze()
            assert img.all() == inv.all()


