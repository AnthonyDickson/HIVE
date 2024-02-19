#  HIVE, creates 3D mesh videos.
#  Copyright (C) 2024 Anthony Dickson anthony.dickson9656@gmail.com
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

import os
from argparse import Namespace, ArgumentParser

import cv2
from lpips import LPIPS

from compare_image_pair import compare_images


def get_arguments() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument('--pairs_path', type=str,
                        help='The path to a folder that contains sub-folders of image pairs.')
    parser.add_argument('--save_latex', action='store_true',
                        help='Whether to save the results in a latex file. '
                             'This option assumes image pair folder names are in the format `<dataset>_<config>`.')

    args = parser.parse_args()

    return args


def main(pairs_path: str, save_latex: bool):
    lpips_fn = LPIPS(net='alex')
    results = dict()

    for pair_name in os.listdir(pairs_path):
        ref_filename, est_filename = os.listdir(os.path.join(pairs_path, pair_name))
        ref_image_path = os.path.join(pairs_path, pair_name, ref_filename)
        est_image_path = os.path.join(pairs_path, pair_name, est_filename)
        ref_image = cv2.imread(ref_image_path)
        est_image = cv2.imread(est_image_path)
        ssim_score, psnr_score, lpips_score = compare_images(ref_image, est_image, lpips_fn)

        results[pair_name] = (ssim_score, psnr_score, lpips_score)

        print(f"{pair_name} - SSIM: {ssim_score:.2f} - PSNR: {psnr_score:.2f} - LPIPS: {lpips_score:.2f}")

    if save_latex:
        latex_lines = [
            r"\begin{tabular}{llrrr}",
            r"\toprule",
            r"Dataset & Config & SSIM & PSNR & LPIPS \\",
            r"\midrule",
        ]

        mean = {
            'ssim': 0.0,
            'psnr': 0.0,
            'lpips': 0.0
        }
        count = 0

        for pair_name, result in results.items():
            parts = pair_name.split('_')
            label = parts[-1]
            name = ' '.join(parts[:-1])
            ssim, psnr, lpips = result
            latex_lines.append(rf"{name} & {label} & {ssim:.2f} & {psnr:.2f} & {lpips:.2f} \\")

            mean['ssim'] += ssim
            mean['psnr'] += psnr
            mean['lpips'] += lpips
            count += 1

        latex_lines.append(r"\midrule")
        latex_lines.append(
            rf"Mean & - & {mean['ssim'] / count:.2f} & {mean['psnr'] / count:.2f} & {mean['lpips'] / count:.2f} \\")

        latex_lines.append(r"\bottomrule")
        latex_lines.append(r"\end{tabular}")

        latex_path = os.path.join(pairs_path, 'table.tex')

        with open(latex_path, 'w') as f:
            f.write('\n'.join(latex_lines))

        print(f"Saved latex table to {latex_path}.")


if __name__ == '__main__':
    args = get_arguments()
    main(pairs_path=args.pairs_path, save_latex=args.save_latex)
