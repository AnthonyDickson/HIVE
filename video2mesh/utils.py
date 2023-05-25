"""This module contains various utility functions that can't be stuffed into other modules."""

import contextlib
import datetime
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
    # TODO: Different files for 'simple' and 'detailed' output?
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    # This line is needed to ensure logging works as expected. Otherwise, logs may be unexpectedly written to certain
    # streams or files.
    logger.handlers = []

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

        def filter(self, record: logging.LogRecord) -> bool:
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

    # TODO: Get rid of matplotlib and trimesh debug output.
    # This line prevents PILLOW from polluting the debug logs.
    logging.getLogger('PIL').setLevel(logging.INFO)


def format_bytes(bytes_count: int) -> str:
    """Format a number of bytes as the appropriate unit (e.g., KiB, MiB, GiB)."""
    for unit in ["", "Ki", "Mi", "Gi", "Ti"]:
        if abs(bytes_count) < 1024.0:
            return f"{bytes_count:3.1f} {unit}B"

        bytes_count /= 1024.0

    return f"{bytes_count:3.1f} PiB"


class Timer:
    """Utility for timing operations. Can be used as a context manager."""

    def __init__(self):
        self._start_time = datetime.datetime.fromtimestamp(0)
        self._stop_time = None

    @property
    def start_time(self) -> datetime.datetime:
        """
        :return: The `datetime` object when the timer was started. See `start()`.
        """
        return self._start_time

    @property
    def stop_time(self) -> datetime.datetime:
        """
        :return: The `datetime` object when the timer was stopped. See `stop()`.
        """
        return self._stop_time

    @property
    def elapsed(self) -> datetime.timedelta:
        """
        :return: The `timedelta` object indicating how much time elapsed. If the timer has been stopped, will return
          the elapsed time between `start_time` and `end_time`, otherwise will return the elapsed time between
          `start_time` and now.
        """
        if self._stop_time is not None:
            return self._stop_time - self._start_time
        else:
            return datetime.datetime.now() - self._start_time

    def start(self):
        """Start the timer."""
        self._start_time = datetime.datetime.now()
        self._stop_time = None

    def stop(self):
        """Stop the timer."""
        self._stop_time = datetime.datetime.now()

    def __enter__(self):
        self.start()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()


def set_key_path(dictionary: dict, path: list, value: Any):
    """
    Set a value in a nested dictionary.

    **Note:** This method modifies the input dictionary in-place.

    >>> d = dict()
    >>> path = ('foo', 'bar', 'baz')
    >>> value = 42
    >>> set_key_path(d, path, value)
    >>> print(d)
    {'foo': {'bar': {'baz': 42}}}

    :param dictionary: A dictionary.
    :param path: A list of dictionary keys.
    :param value: The value to set the nested dictionary entry to.
    """
    dict_entry = dictionary

    for key in path:
        if key not in dict_entry:
            dict_entry[key] = dict()

        if key == path[-1]:
            dict_entry[key] = value
        else:
            dict_entry = dict_entry[key]


def get_key_path(dictionary: dict, path: list) -> Any:
    """
    Get a value from a nested dictionary.

    **Note:** This method modifies the input dictionary in-place.

    >>> d =  {'foo': {'bar': {'baz': 42}}}
    >>> get_key_path(['foo', 'bar', 'baz'])
    42

    :param dictionary: A dictionary.
    :param path: A list of dictionary keys.
    :return: The value at the given path.
    """
    dict_entry = dictionary

    for key in path:
        if key == path[-1]:
            return dict_entry[key]

        dict_entry = dict_entry[key]


@contextlib.contextmanager
def timed_block(log_msg: Optional[str], profiling: Optional[dict], key_path: list):
    """
    Log a message, run a block of code, and write the runtime of the block to `self.profiling`.

    :param profiling: A dictionary for recording runtime statistics.
    :param log_msg: The optional message to log.
    :param key_path: The dictionary path(s) to write the runtime to, e.g. ['my_app', 'total_runtime'].
        Any nested dictionaries or keys that do not exist will be created automatically.
    """
    if log_msg:
        logging.info(log_msg)

    timer = Timer()
    timer.start()

    try:
        yield timer
    finally:
        if profiling is None:
            return

        set_key_path(profiling, key_path, timer.elapsed.total_seconds())
