import os
import random
from pathlib import Path

import numpy as np
import torch

from dpipe.io import PathLike
import surface_distance.metrics as surf_dc

from functools import wraps, partial
from typing import Union, Callable

import numpy as np

from dpipe.im.axes import broadcast_to_axes, AxesLike, AxesParams
from dpipe.im.grid import divide, combine
from dpipe.itertools import extract
from dpipe.im.shape_ops import pad_to_shape, crop_to_shape, pad_to_divisible
from dpipe.im.shape_utils import prepend_dims, extract_dims
from dpipe.itertools import pmap


def get_pred(x, threshold=0.5):
    return x > threshold


def get_damri_dir_name():
    return os.path.dirname(__file__)


def fix_seed(seed=0xBadCafe):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def choose_root(*paths: PathLike) -> Path:
    for path in paths:
        path = Path(path)
        if path.exists():
            return path
    raise FileNotFoundError('No appropriate root found.')


def sdice(a, b, spacing,tolerance):
    surface_distances = surf_dc.compute_surface_distances(a, b, spacing)
    return surf_dc.compute_surface_dice_at_tolerance(surface_distances, tolerance)


def skip_predict(output_path):
    print(f'>>> Passing the step of saving predictions into `{output_path}`', flush=True)
    os.makedirs(output_path)

# The same as in the shape.py file in dpipe, but for the DANN input format
# Net(x,y)

def add_extract_dims(n_add: int = 1, n_extract: int = None, sequence: bool = False):
    """
    Adds ``n_add`` dimensions before a prediction and extracts ``n_extract`` dimensions after this prediction.

    Parameters
    ----------
    n_add: int
        number of dimensions to add.
    n_extract: int, None, optional
        number of dimensions to extract. If ``None``, extracts the same number of dimensions as were added (``n_add``).
    sequence:
        if True - the output is expected to be a sequence, and the dims are extracted for each element of the sequence.
    """
    if n_extract is None:
        n_extract = n_add

    def decorator(predict):
        @wraps(predict)
        def wrapper(*xs, **kwargs):
            result = predict(*[prepend_dims(x, n_add) for x in xs], **kwargs)
            if sequence:
                return [extract_dims(entry, n_extract) for entry in result]

            return extract_dims(result, n_extract)

        return wrapper

    return decorator


def divisible_shape(divisor: AxesLike, axes: AxesLike = None, padding_values: Union[AxesParams, Callable] = 0,
                    ratio: AxesParams = 0.5):
    """
    Pads an incoming array to be divisible by ``divisor`` along the ``axes``. Afterwards the padding is removed.

    Parameters
    ----------
    divisor
        a value an incoming array should be divisible by.
    axes
        axes along which the array will be padded. If None - the last ``len(divisor)`` axes are used.
    padding_values
        values to pad with. If Callable (e.g. ``numpy.min``) - ``padding_values(x)`` will be used.
    ratio
        the fraction of the padding that will be applied to the left, ``1 - ratio`` will be applied to the right.

    References
    ----------
    `pad_to_divisible`
    """
    axes, divisor, ratio = broadcast_to_axes(axes, divisor, ratio)

    def decorator(predict):
        @wraps(predict)
        def wrapper(x1, x2, *args, **kwargs):
            shape = np.array(x1.shape)[list(axes)]
            x1 = pad_to_divisible(x1, divisor, axes, padding_values, ratio)
            x2 = pad_to_divisible(x2, divisor, axes, padding_values, ratio)
            result = predict(x1, x2, *args, **kwargs)
            return crop_to_shape(result, shape, axes, ratio)

        return wrapper

    return decorator


def patches_grid(patch_size: AxesLike, stride: AxesLike, axes: AxesLike = None,
                 padding_values: Union[AxesParams, Callable] = 0, ratio: AxesParams = 0.5):
    """
    Divide an incoming array into patches of corresponding ``patch_size`` and ``stride`` and then combine
    predicted patches by averaging the overlapping regions.

    If ``padding_values`` is not None, the array will be padded to an appropriate shape to make a valid division.
    Afterwards the padding is removed.

    References
    ----------
    `grid.divide`, `grid.combine`, `pad_to_shape`
    """
    axes, patch_size, stride = broadcast_to_axes(axes, patch_size, stride)
    valid = padding_values is not None

    def decorator(predict):
        @wraps(predict)
        def wrapper(x1, x2, *args, **kwargs):
            if valid:
                shape = np.array(x1.shape)[list(axes)]
                padded_shape = np.maximum(shape, patch_size)
                new_shape = padded_shape + (stride - padded_shape + patch_size) % stride
                x1 = pad_to_shape(x1, new_shape, axes, padding_values, ratio)
                x2 = pad_to_shape(x2, new_shape, axes, padding_values, ratio)

            patches = pmap(predict, divide(x1, patch_size, stride, axes), *args, **kwargs)
            prediction1 = combine(patches, extract(x1.shape, axes), stride, axes)
            if valid:
                prediction1 = crop_to_shape(prediction1, shape, axes, ratio)

            patches = pmap(predict, divide(x2, patch_size, stride, axes), *args, **kwargs)
            prediction2 = combine(patches, extract(x2.shape, axes), stride, axes)
            if valid:
                prediction2 = crop_to_shape(prediction1, shape, axes, ratio)

            return prediction1, prediction2

        return wrapper

    return decorator
