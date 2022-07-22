"""
This module contains functions for transforming data stored as images such as depth maps and instance segmentation
masks.
"""
import logging

import cv2
import numpy as np

from video2mesh.options import MaskDilationOptions
from video2mesh.utils import validate_shape


def dilate_mask(mask, dilation_options: MaskDilationOptions):
    """
    Dilate an instance segmentation mask so that it covers a larger area.

    :param mask: The mask to enlarge/dilate.
    :param dilation_options: The object containing the dilation mask/filter and other settings.

    :return: The dilated mask.
    """
    validate_shape(mask, 'mask', expected_shape=(None, None))

    mask = mask.astype(np.float32)
    mask = cv2.dilate(mask.astype(float), dilation_options.filter, iterations=dilation_options.num_iterations)
    mask = mask.astype(bool)

    return mask


def median_filter(depth_map: np.ndarray, kernel_size=63) -> np.ndarray:
    """
    Apply a median filter to a depth map.
    :param depth_map: The depth map to filter in NYU format (invalid = 0, max = 10.0).
    :param kernel_size: The height and width of the filter. If the filter is larger than 5 pixels, the depth values
        will first be converted to 8-bit values, filtered then converted back to float values.
    :return: The filtered depth map.
    """
    if kernel_size > 5:
        # noinspection PyArgumentList
        min_depth = depth_map.min()
        # noinspection PyArgumentList
        max_depth = depth_map.max()

        depth_map = (depth_map - min_depth) / (max_depth - min_depth)
        depth_map = (255 * depth_map).astype(np.uint8)

        filtered = cv2.medianBlur(depth_map, kernel_size)

        filtered_depth = (filtered / 255).astype(np.float32)
        filtered_depth = filtered_depth * (max_depth - min_depth) + min_depth

        return filtered_depth
    else:
        return cv2.medianBlur(depth_map, kernel_size)


def get_bins(min_value=0.0, max_value=10.0, num_bins=128) -> np.ndarray:
    """
    Get bins for the specified range according to the approach described in the paper "Deep Ordinal Regression Network
    for Monocular Depth Estimation" https://arxiv.org/abs/1806.02446.

    :param min_value: The lower bound.
    :param max_value: The upper bound.
    :param num_bins: The number bins to separate the input space into.
    :return: The list of bin thresholds
    """
    epsilon = 1.0 - min_value
    alpha_ = min_value + epsilon
    beta_ = max_value + epsilon
    thresholds = [np.power(np.e, np.log(alpha_) + (np.log(beta_ / alpha_) * i) / num_bins) - epsilon
                  for i in range(num_bins)]

    return np.asarray(thresholds + [max_value])


def bin_depth(depth_map, min_depth=0.0, max_depth=10.0, num_bins=128):
    """
    Quantize depth values into ordered bins of increasing size.

    :param depth_map: The depth map to quantize.
    :param min_depth: The smallest depth value allowed in the depth map.
    :param max_depth: The largest depth value allowed in the depth map.
    :param num_bins: The number of times to divide the range of depth values.
    :return: The depth map using the binned depth values.
    """
    bins = get_bins(min_depth, max_depth, num_bins=num_bins)
    # Digitize returns the bin indices.
    depth_bins = np.digitize(depth_map, bins)
    # To get the binned value (the bin's lower bound), just index the bins array with the (H, W) map of bin indices.
    binned_depth = bins[depth_bins]

    return binned_depth


def calculate_target_resolution(source_hw, target_hw):
    """
    Calculate the target resolution and perform some sanity checks.

    :param source_hw: The resolution of the input frames. These are used if the target resolution is given as a
        single value indicating the desired length of the longest side of the images.
    :param target_hw: The resolution (height, width) to resize the images to.
    :return: The target resolution as a 2-tuple (height, width).
    """
    if isinstance(target_hw, int):
        # Cast results to int to avoid warning highlights in IDE.
        longest_side = int(np.argmax(source_hw))
        shortest_side = int(np.argmin(source_hw))

        new_size = [0, 0]
        new_size[longest_side] = target_hw

        scale_factor = new_size[longest_side] / source_hw[longest_side]
        new_size[shortest_side] = int(source_hw[shortest_side] * scale_factor)

        target_hw = new_size
    elif isinstance(target_hw, tuple):
        if len(target_hw) != 2:
            raise ValueError(f"The target resolution must be a 2-tuple, but got a {len(target_hw)}-tuple.")

        if not isinstance(target_hw[0], int) or not isinstance(target_hw[1], int):
            raise ValueError(f"Expected target resolution to be a 2-tuple of integers, but got a tuple of"
                             f" ({type(target_hw[0])}, {type(target_hw[1])}).")

    target_orientation = 'portrait' if np.argmax(target_hw) == 0 else 'landscape'
    source_orientation = 'portrait' if np.argmax(source_hw) == 0 else 'landscape'

    if target_orientation != source_orientation:
        logging.warning(
            f"The input images appear to be in {source_orientation} ({source_hw[1]}x{source_hw[0]}), "
            f"but they are being resized to what appears to be "
            f"{target_orientation} ({target_hw[1]}x{target_hw[0]})")

    source_aspect = np.round(source_hw[1] / source_hw[0], decimals=2)
    target_aspect = np.round(target_hw[1] / target_hw[0], decimals=2)

    if not np.isclose(source_aspect, target_aspect):
        logging.warning(f"The aspect ratio of the source video is {source_aspect:.2f}, "
                        f"however the aspect ratio of the target resolution is {target_aspect:.2f}. "
                        f"This may lead to stretching in the images.")

    return target_hw
