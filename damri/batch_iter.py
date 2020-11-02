import random

import numpy as np

from dpipe.im.patch import sample_box_center_uniformly
from dpipe.im.box import get_centered_box
from dpipe.im.shape_ops import crop_to_box
from dpipe.im.slices import iterate_slices
from dpipe.itertools import lmap
from dpipe.batch_iter import unpack_args

# import for load_by_random_id_DANN
from typing import Sequence, Callable, Union
import numpy as np
from dpipe.itertools import pam, squeeze_first
from dpipe.batch_iter.sources import sample


SPATIAL_DIMS = (-3, -2, -1)


def get_random_slice(*arrays, interval: int = 1):
    slc = np.random.randint(arrays[0].shape[-1] // interval) * interval
    return tuple(array[..., slc] for array in arrays)


def get_only_one_slice(*arrays):
    return tuple(array[..., arrays[0].shape[-1] // 2] for array in arrays)


def sample_center_uniformly(shape, patch_size, spatial_dims):
    spatial_shape = np.array(shape)[list(spatial_dims)]
    if np.all(patch_size <= spatial_shape):
        return sample_box_center_uniformly(shape=spatial_shape, box_size=patch_size)
    else:
        return spatial_shape // 2


def center_choice_random(inputs, y_patch_size):
    x, y = inputs
    center = sample_center_uniformly(y.shape, patch_size=y_patch_size, spatial_dims=SPATIAL_DIMS)
    return x, y, center


def center_choice_random_DANN(inputs, y_patch_size):
    image_l, image_u, mask_l, label_l, label_u = inputs
    center = sample_center_uniformly(mask_l.shape, patch_size=y_patch_size, spatial_dims=SPATIAL_DIMS)
    return image_l, image_u, mask_l, label_l, label_u, center


def slicewise(predict):
    def wrapper(*arrays):
        return np.stack(lmap(unpack_args(predict), iterate_slices(*arrays, axis=-1)), -1)

    return wrapper

def load_by_random_id_LU(*loaders: Callable, ids_l: Sequence, ids_u: Sequence,
                      weights: Sequence[float] = None,
                      random_state: Union[np.random.RandomState, int] = None):

    # load_by_random_id for Labelled (L) and Unlabelled (U) scans loading
    for id_l, id_u in zip(sample(ids_l, weights, random_state),
                          sample(ids_u, weights, random_state)):

        output = squeeze_first(tuple(pam(loaders, id_l)) +\
                            tuple(pam(loaders[:-1], id_u)))
        # I believe, that we do not really need squeeze_first, but let it be as for now
        # It doesn't hurt anyway and maybe I did not take smth into account
        order = [0,3,2,1,4]
        outputPermuted = ()
        for i in order:
            outputPermuted += (output[i],)

        yield outputPermuted

def center_choice_ts(inputs, y_patch_size, nonzero_fraction=0.5, tumor_sampling=True, with_cc=False):
    if with_cc:
        x, y, cc, centers = inputs
    else:
        x, y, centers = inputs

    y_patch_size = np.array(y_patch_size)
    if len(centers) > 0 and np.random.uniform() < nonzero_fraction:
        center = random.choice(random.choice(centers)) if tumor_sampling else random.choice(centers)
    else:
        center = sample_center_uniformly(y.shape, patch_size=y_patch_size, spatial_dims=SPATIAL_DIMS)

    if with_cc:
        return x, y, cc, center
    else:
        return x, y, center


def extract_patch(inputs, x_patch_size, y_patch_size):
    x, y, center = inputs

    x_patch_size = np.array(x_patch_size)
    y_patch_size = np.array(y_patch_size)
    x_spatial_box = get_centered_box(center, x_patch_size)
    y_spatial_box = get_centered_box(center, y_patch_size)

    x_patch = crop_to_box(x, box=x_spatial_box, padding_values=np.min)
    y_patch = crop_to_box(y, box=y_spatial_box, padding_values=0)
    return x_patch, y_patch


def extract_patch_DANN(inputs, x_patch_size, y_patch_size):
    image_l, image_u, mask_l, label_l, label_u, center = inputs

    x_patch_size = np.array(x_patch_size)
    y_patch_size = np.array(y_patch_size)
    x_spatial_box = get_centered_box(center, x_patch_size)
    y_spatial_box = get_centered_box(center, y_patch_size)

    image_l_patch = crop_to_box(image_l, box=x_spatial_box, padding_values=np.min)
    image_u_patch = crop_to_box(image_u, box=x_spatial_box, padding_values=np.min)
    mask_l_patch = crop_to_box(mask_l, box=y_spatial_box, padding_values=0)

    return image_l_patch, image_u_patch, mask_l_patch, label_l, label_u


def flip_augm(inputs, p=0.3, dims=None):
    if dims is None:
        dims = SPATIAL_DIMS
    outputs = inputs
    for dim in dims:
        if np.random.rand() < p:
            outputs = (np.flip(x, axis=dim) for x in inputs)
    return outputs


def rot_augm(inputs, p=0.3, dims=None):
    if dims is None:
        dims = SPATIAL_DIMS
    outputs = inputs
    if np.random.rand() < p:
        axes = tuple(np.random.permutation(dims)[:2])
        k = np.random.randint(1, 4)
        outputs = (np.rot90(x, k=k, axes=axes) for x in inputs)
    return outputs


def augm_spatial(inputs, dims_flip=None, dims_rot=None):
    return rot_augm(flip_augm(inputs, dims=dims_flip), dims=dims_rot)
