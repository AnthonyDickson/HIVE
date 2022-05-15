"""
This module contains functions for transforming data stored as images such as depth maps and instance segmentation
masks.
"""

import cv2
import numpy as np

from Video2mesh.options import MaskDilationOptions
from Video2mesh.utils import validate_shape


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
