import datetime
from typing import Optional

import numpy as np
import sys


def validate_camera_parameter_shapes(K, R, t):
    validate_shape(K, 'K', expected_shape=(3, 3))
    validate_shape(R, 'R', expected_shape=(3, 3))
    validate_shape(t, 't', expected_shape=(3, 1))


def validate_shape(x: np.ndarray, x_name: str, expected_shape: tuple):
    """
    Validate (assert) that the shape of a array matches the expected shape, otherwise raise a descriptive AssertionError.

    :param x: The array to validate.
    :param x_name: The name of the array (e.g. the parameter name).
    :param expected_shape: The expected shape as a tuple. Dimensions with a variable size can indicated with a
    value of None, e.g. (None, 3) accepts any 2D array as long as the second dimension has a size of 3.
    """
    assert type(expected_shape) is tuple, "`expected_shape` must be a tuple."

    expected_dims = len(expected_shape)
    observed_dims = len(x.shape)

    assert observed_dims == expected_dims, \
        f"Incorrect number of dimensions for {x_name}; expected {expected_dims} but got {observed_dims}"

    dim_matches = []

    for dim in range(expected_dims):
        dim_matches.append(expected_shape[dim] is None or x.shape[dim] == expected_shape[dim])

    expected_shape_str = f"({', '.join(map(num2str, expected_shape))})"

    assert np.alltrue(dim_matches), \
        f"Incorrect shape for {x_name}: expected {expected_shape_str} but got {x.shape}"


def num2str(num: Optional[int]):
    """
    Convert an optional number (i.e. nullable) to a string.

    :param num: the number to convert.

    :return: the number as a string, '?' if the the argument is None.
    """
    return '?' if num is None else str(num)


def log(message, prefix='', end='\n', file=None):
    print(f"{prefix}[{datetime.datetime.now().time()}] {message}", file=file, end=end)
