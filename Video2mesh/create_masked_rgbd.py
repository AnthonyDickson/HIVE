import cv2
import datetime
import os.path

import imageio
import numpy as np
import psutil
from multiprocessing.pool import ThreadPool

import argparse

from Video2mesh.io import VTMDataset
from Video2mesh.utils import log

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', type=str, help='Path to the RGB-D dataset.')

    args = parser.parse_args()

    start = datetime.datetime.now()

    dataset_path = args.dataset_path

    VTMDataset._validate_dataset(dataset_path)

    dataset = VTMDataset(dataset_path) \
        .create_or_find_masks()

    log("Load dataset")

    masked_depth_folder = 'masked_depth'
    masked_depth_path = os.path.join(dataset.base_path, masked_depth_folder)

    os.makedirs(masked_depth_path, exist_ok=True)
    log("Create output folder")

    pool = ThreadPool(processes=psutil.cpu_count(logical=False))
    log("Create thread pool")

    dilation_filter = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))


    def save_depth(i, depth_map, mask):
        binary_mask = mask > 0.0
        binary_mask = binary_mask.astype(np.float32)
        binary_mask = cv2.dilate(binary_mask.astype(float), dilation_filter, iterations=64)
        binary_mask = binary_mask.astype(bool)

        depth_map[binary_mask] = 0.0
        depth_map = depth_map / dataset.depth_scaling_factor  # undo depth scaling done during loading.
        depth_map = depth_map.astype(np.uint16)
        output_path = os.path.join(masked_depth_path, f"{i:06d}.png")
        imageio.imwrite(output_path, depth_map)

        log(f"Writing masked depth to {output_path}")


    pool.starmap(save_depth, zip(range(len(dataset)), dataset.depth_dataset, dataset.mask_dataset))

    elapsed = datetime.datetime.now() - start

    log(f"Done in {elapsed}")
