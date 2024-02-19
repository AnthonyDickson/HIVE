"""This program compares two images with the SSIM, PSNR and LIPIPS metrics. Note that it assumes that the images are both: 8-bit, 3-channel (no alpha channel) and have the same resolution."""

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
    

def get_arguments() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument('--ref_image', type=str, help='The path to the reference image.')
    parser.add_argument('--est_image', type=str, help='The path to the estimated image.')

    args = parser.parse_args()

    return args


def compare_images(ref_image, est_image, lpips_fn=None):
    """
    Calculate image similarity between two images.

    :param ref_image: An image loaded via OpenCV (`cv2.imread`) to compare against.
    :param est_image: An image loaded via OpenCV (`cv2.imread`) to compare against the reference image.
    :param lpips_fn: The LPIPS function to use. If `None`, the default LPIPS function is used (AlexNet).
    :return: A 3-tuple containing the structural similarity, peak-signal-to-noise ratio, and the LPIPS scores.
    """
    ref_gray = cv2.cvtColor(ref_image, cv2.COLOR_BGR2GRAY)
    est_gray = cv2.cvtColor(est_image, cv2.COLOR_BGR2GRAY)

    ssim_score = structural_similarity(ref_gray, est_gray, win_size=7)
    psnr_score = peak_signal_noise_ratio(ref_gray, est_gray)

    lpips_fn = LPIPS(net='alex') if lpips_fn is None else lpips_fn
    lpips_score = measure_lpips(ref_image, est_image, lpips_fn)

    return ssim_score, psnr_score, lpips_score


def main(ref_image: str, est_image: str):
    ref_image = cv2.imread(ref_image)
    est_image = cv2.imread(est_image)
    lpips_fn = LPIPS(net='alex')
    ssim_score, psnr_score, lpips_score = compare_images(ref_image, est_image, lpips_fn)
    print(f"SSIM: {ssim_score:,.2f} - PSNR: {psnr_score:,.2f} dB - LPIPS: {lpips_score:,.2f}")


if __name__ == '__main__':
    args = get_arguments()
    main(ref_image=args.ref_image, est_image=args.est_image)
