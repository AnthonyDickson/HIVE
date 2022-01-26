import datetime

import sys

import time
from typing import Optional

import numpy as np


class Timer:
    def __init__(self):
        self.start_time = time.time()
        self.stop_time = time.time()

        self.splits = [0.0]
        self.split_names = ['start']

    def start(self):
        self.start_time = time.time()

    def split(self, split_name: Optional[str] = None, verbose=True):
        elapsed = time.time() - self.start_time
        self.splits.append(elapsed)
        self.split_names.append(split_name or '')

        if verbose:
            print(self.split_to_string(i=-1))

        return elapsed

    def split_to_string(self, i=-1):
        elapsed = self.splits[i]
        split_name = self.split_names[i]

        elapsed_precision = 2 if elapsed > 1.0 else 3
        split_string = f"{split_name.capitalize()}: {elapsed:,.{elapsed_precision}f}"

        if len(self.splits) > 1 and i != 0:
            prev_time = self.splits[i - 1]

            delta = elapsed - prev_time
            delta_precision = 2 if delta > 1.0 else 3

            split_string = f"{split_string} (+{delta:,.{delta_precision}f})"

        return split_string

    def stop(self):
        self.stop_time = self.start_time + self.split('stop')

    def __enter__(self):
        self.start()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()


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


def log(message, prefix='', end='\n', file=sys.stdout):
    print(f"{prefix}[{datetime.datetime.now().time()}] {message}", file=file, end=end)