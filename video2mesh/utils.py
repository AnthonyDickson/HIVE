"""This module contains various utility functions that can't be stuffed into other modules."""

import contextlib
import enum
import logging
import sys
from multiprocessing.pool import ThreadPool
from typing import Optional, Any, Union, Type

import numpy as np
import psutil
import torch
from tqdm import tqdm


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

    :return: the number as a string, '?' if the argument is None.
    """
    return '?' if num is None else str(num)


def tqdm_imap(func, args, num_processes: Optional[int] = None) -> list:
    """
    Process args in parallel with a progress bar.
    Note that this function blocks and the results are not accessible until all `args` have been processed.

    >>> def f(i):
    >>>     return i ** 2
    >>> result = tqdm_imap(f, range(10))
    >>> print(result)
    [0, 1, 4, 9, 16, 25, 36, 49, 64, 81]

    :param func: The function to apply to the given arguments.
    :param args: List of arguments to proces.
    :param num_processes: (optional) Number of processes to spawn.
    :return: A list of the values returned by `func`.
    """
    pool = ThreadPool(processes=num_processes or psutil.cpu_count())

    results = []

    # Not sure why, but the below line does not work as expected.
    # It does not block, and you need to call pool.stop() -> pool.join() to force Python to wait until all jobs are
    # completed. However, this introduces a new problem where the progress bar gets stuck at 0.
    # There's something about looping over the tqdm wrapped iterable (perhaps the call to next(...)) that fixes this.
    # tqdm(pool.imap(save_depth_wrapper, args), total=len(args))
    for return_value in tqdm(pool.imap(func, args), total=len(args)):
        results.append(return_value)

    return results


@contextlib.contextmanager
def temp_seed(seed):
    """Seed NumPy with a temporary seed."""
    state = np.random.get_state()

    try:
        np.random.seed(seed)

        yield
    finally:
        np.random.set_state(state)


@contextlib.contextmanager
def cudnn():
    """Temporarily enable the cuDNN backend for torch."""
    cudnn_enabled = torch.backends.cudnn.enabled
    cudnn_benchmark = torch.backends.cudnn.benchmark

    try:
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True

        yield
    finally:
        torch.backends.cudnn.enabled = cudnn_enabled
        torch.backends.cudnn.benchmark = cudnn_benchmark


# noinspection PyArgumentList
class Domain(enum.Enum):
    """The domain of a value (positive, non-negative or negative)."""

    # < 0
    Negative = enum.auto()
    # > 0
    Positive = enum.auto()
    # >= 0
    NonNegative = enum.auto()


def check_domain(value: Any, name: str, value_type: Union[Type[int], Type[float]], domain: Optional[Domain] = None,
                 nullable=False):
    """
    Check whether the given value is within the specified domain, e.g. whether the variable `x` is a positive integer.
    Raises `ValueError` if the value is not in the specified domain.

    :param value: The value to check.
    :param name: The name of the variable/value that is being checked. Used in the exception text.
    :param value_type: Should the value be an integer or float?
    :param domain: Should the value be positive, non-negative or negative?
    :param nullable: Can this value be `None`?

    :raises: ValueError if the value is outside the specified domain.
    """
    if nullable and value is None:
        return

    if domain is not None:
        domain_name = f" {domain.name.lower()} "

        if domain == Domain.Negative:
            in_domain = value < 0.0
        elif domain == Domain.NonNegative:
            in_domain = value >= 0.0
            domain_name = ' non-negative '
        elif domain == Domain.Positive:
            in_domain = value > 0.0
        else:
            raise RuntimeError(f"Unsupported domain type {domain}.")
    else:
        domain_name = ''
        in_domain = True

    if not isinstance(value, value_type) or not in_domain:
        raise ValueError(f"{name} must be a {domain_name}{value_type}, but got {value} ({type(value)}) instead")


def setup_logger(log_path: Optional[str] = None):
    """
    Configure the logger.

    :param log_path: (optional) The file to save the logs to. If set to None, logs will not be written to disk.
    """
    # TODO: Get rid of logs from installed packages from DEBUG level output.
    # TODO: Different files for 'simple' and 'detailed' output?
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    detailed_formatter = logging.Formatter(
        '[%(asctime)s.%(msecs)03d] [%(levelname)s] %(pathname)s:%(lineno)s: %(funcName)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    simple_formatter = logging.Formatter(
        '[%(asctime)s.%(msecs)03d] [%(levelname)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    if log_path is not None:
        logging_file_handler = logging.FileHandler(log_path)
        logging_file_handler.setLevel(logging.DEBUG)
        logging_file_handler.setFormatter(detailed_formatter)
        logger.addHandler(logging_file_handler)

    class IgnoreLevelsAbove(logging.Filter):
        def __init__(self, level, name=''):
            super().__init__(name=name)

            self.__level = level

        def filter(self,  record: logging.LogRecord) -> bool:
            return record.levelno <= self.__level

    # All messages that are DEBUG or INFO will be captured by this handler and printed to stdout.
    logging_stdout_handler = logging.StreamHandler(sys.stdout)
    logging_stdout_handler.setLevel(logging.INFO)
    logging_stdout_handler.addFilter(IgnoreLevelsAbove(logging.INFO))
    logging_stdout_handler.setFormatter(simple_formatter)
    logger.addHandler(logging_stdout_handler)

    # This handler handles all messages that are >= WARNING and print them to stderr.
    logging_stderr_handler = logging.StreamHandler(sys.stderr)
    logging_stderr_handler.setLevel(logging.WARNING)
    logging_stderr_handler.setFormatter(detailed_formatter)
    logger.addHandler(logging_stderr_handler)
