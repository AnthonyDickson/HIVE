"""This program compares two images with the SSIM, PSNR, LPIPS and MIFD metrics. Note that it assumes that the images are both: 8-bit, 3-channel (no alpha channel) and have the same resolution."""

#  HIVE, creates 3D mesh videos.
#  Copyright (C) 2023 Anthony Dickson anthony.dickson9656@gmail.com
#
#  This program is free software: you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License
#  along with this program.  If not, see <https://www.gnu.org/licenses/>.

import warnings
from argparse import Namespace, ArgumentParser

import cv2
import numpy as np
import torch
from lpips import LPIPS
from skimage.metrics import structural_similarity, peak_signal_noise_ratio


def measure_lpips(reference_image, comparison_image, lpips_fn):
    reference_image = (reference_image / 255) * 2.0 - 1.0
    reference_image = reference_image.transpose((2, 0, 1))
    reference_image = np.expand_dims(reference_image, axis=0)
    reference_image = torch.from_numpy(reference_image).to(torch.float32)

    comparison_image = (comparison_image / 255) * 2.0 - 1.0
    comparison_image = comparison_image.transpose((2, 0, 1))
    comparison_image = np.expand_dims(comparison_image, axis=0)
    comparison_image = torch.from_numpy(comparison_image).to(torch.float32)

    with torch.no_grad():
        return lpips_fn.forward(reference_image, comparison_image).item()


def mifd(label: np.ndarray, output: np.ndarray, ratio_threshold=0.7, k=2, min_matches=1, log_residual=False):
    """
    Calculate the MIFD (Mean Image Feature Distance) metric between two grayscale images.

    :param label: The reference grayscale image.
    :param output: The 'estimated' grayscale image.
    :param ratio_threshold: The threshold to use for Lowe's ratio test for filtering matches.
    :param k: The number of matches to consider in the KNN matcher.
    :param min_matches: The minimum allowable number of correspondences. Image pairs with fewer correspondences are
        assigned 'nan'.
    :param log_residual: Whether to measure distances in log10 space.
    :return: The MIFD score (potentially 'nan').
    """
    # -- Step 1: Detect the keypoints using SURF Detector, compute the descriptors
    detector = cv2.SIFT.create()
    # detector = cv2.xfeatures2d.SIFT_create()
    key_points1, descriptors1 = detector.detectAndCompute(label, None)
    key_points2, descriptors2 = detector.detectAndCompute(output, None)
    # -- Step 2: Matching descriptor vectors with a FLANN based matcher
    if descriptors1 is None or descriptors2 is None:
        warnings.warn(
            f"Could not extract any features for at least one image in the pair.")
        return float('nan')

    if len(descriptors1) < k or len(descriptors2) < k:
        warnings.warn(
            f"Not enough descriptors for k={k:d}, only got {len(descriptors1):,d} and {len(descriptors2):,d}.")
        return float('nan')

    # Since SURF is a floating-point descriptor NORM_L2 is used
    matcher = cv2.DescriptorMatcher_create(cv2.DescriptorMatcher_FLANNBASED)
    knn_matches = matcher.knnMatch(descriptors1, descriptors2, k)
    # -- Filter matched key points using the Lowe's ratio test
    points1 = []
    points2 = []

    for m, n in knn_matches:
        if m.distance < ratio_threshold * n.distance:
            points1.append(key_points1[m.queryIdx].pt)
            points2.append(key_points2[m.trainIdx].pt)

    if len(points1) < min_matches:
        warnings.warn(f"Not enough matches for `min_matches={min_matches}`, only got {len(points1)}.")
        return float('nan')

    if log_residual:
        residuals = np.log10(points1) - np.log10(points2)
    else:
        residuals = np.asarray(points1) - np.asarray(points2)

    try:
        return np.mean(np.sqrt(np.sum(np.square(residuals), axis=1)))
    except np.AxisError:  # No matches means axis 1 will be out of bounds.
        return float('nan')
    

def get_arguments() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument('--ref_image', type=str, help='The path to the reference image.')
    parser.add_argument('--est_image', type=str, help='The path to the estimated image.')

    args = parser.parse_args()

    return args


def compare_images(ref_image, est_image, lpips_fn=None, return_mifd=False):
    """
    Calculate image similarity between two images.

    :param ref_image: An image loaded via OpenCV (`cv2.imread`) to compare against.
    :param est_image: An image loaded via OpenCV (`cv2.imread`) to compare against the reference image.
    :param lpips_fn: The LPIPS function to use. If `None`, the default LPIPS function is used (AlexNet).
    :param return_mifd: Whether to calculate and return the MIFD metric.
    :return: A 4-tuple containing the structural similarity, peak-signal-to-noise ratio, LPIPS and (optionally) the MIFD scores.
    """
    ref_gray = cv2.cvtColor(ref_image, cv2.COLOR_BGR2GRAY)
    est_gray = cv2.cvtColor(est_image, cv2.COLOR_BGR2GRAY)

    ssim_score = structural_similarity(ref_gray, est_gray, win_size=7)
    psnr_score = peak_signal_noise_ratio(ref_gray, est_gray)

    lpips_fn = LPIPS(net='alex') if lpips_fn is None else lpips_fn
    lpips_score = measure_lpips(ref_image, est_image, lpips_fn)

    if return_mifd:
        mifd_score = mifd(ref_gray, est_gray)
        return ssim_score, psnr_score, lpips_score, mifd_score
    else:
        return ssim_score, psnr_score, lpips_score


def main(ref_image: str, est_image: str):
    ref_image = cv2.imread(ref_image)
    est_image = cv2.imread(est_image)
    lpips_fn = LPIPS(net='alex')
    ssim_score, psnr_score, lpips_score, mifd_score = compare_images(ref_image, est_image, lpips_fn)
    print(f"SSIM: {ssim_score:,.3f} - PSNR: {psnr_score:,.1f} dB - LPIPS: {lpips_score:,.3f} - MIFD: {mifd_score:,.1f}")


if __name__ == '__main__':
    args = get_arguments()
    main(ref_image=args.ref_image, est_image=args.est_image)
