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

import argparse
import json
import logging
import math
import os.path
import shlex
import shutil
import statistics
import subprocess
import urllib.request
import zipfile
from contextlib import contextmanager
from dataclasses import dataclass
from os.path import join as pjoin
from pathlib import Path
from typing import Tuple, Dict, Optional, List, Union, Callable

import cv2
import imageio.v3
import numpy as np
import pandas as pd
import pyrender
import torch
import trimesh
import yaml
from PIL import Image
from lpips import LPIPS
from omegaconf import OmegaConf
from torch.utils.data.dataloader import default_collate

from hive.custom_types import File, Size
from hive.dataset_adaptors import VideoAdaptorBase, estimate_depth_dpt, DatasetAdaptor
from hive.fusion import tsdf_fusion, bundle_fusion
from hive.geometric import Trajectory, pose_vec2mat, CameraMatrix, pose_mat2vec
from hive.io import HiveDataset, temporary_trajectory, DatasetMetadata, COLMAPProcessor, ImageFolderDataset
from hive.options import BackgroundMeshOptions, PipelineOptions, StorageOptions, InpaintingMode, COLMAPOptions, \
    WebXROptions
from hive.pipeline import Pipeline
from hive.sensor import KinectSensor
from hive.utils import setup_logger, tqdm_imap, set_key_path
from scripts.compare_image_pair import compare_images
from third_party.lama.saicinpainting.evaluation.data import pad_img_to_modulo
from third_party.lama.saicinpainting.evaluation.utils import move_to_device
from third_party.lama.saicinpainting.training.trainers import load_checkpoint


def run_trajectory_comparisons(dataset, pred_trajectory: Trajectory, gt_trajectory: Trajectory, dataset_name: str,
                               pred_label: str, gt_label: str, results_dict: dict, output_folder: str,
                               background_mesh_options: BackgroundMeshOptions, frame_set: List[int]):
    experiment_path = pjoin(output_folder, dataset_name, pred_label)
    os.makedirs(experiment_path, exist_ok=True)

    ate = gt_trajectory.calculate_ate(pred_trajectory)
    error_r, error_t = gt_trajectory.calculate_rpe(pred_trajectory)

    gt_trajectory.save(pjoin(experiment_path, 'gt_trajectory.txt'))
    pred_trajectory.save(pjoin(experiment_path, 'pred_trajectory.txt'))

    gt_trajectory.plot(output_path=pjoin(experiment_path, 'gt_trajectory.png'))
    pred_trajectory.plot(output_path=pjoin(experiment_path, 'pred_trajectory.png'))
    gt_trajectory.plot_comparison(pred_trajectory, output_path=pjoin(experiment_path, 'trajectory_comparison.png'))

    # noinspection PyTypeChecker
    np.savetxt(pjoin(experiment_path, 'ate.txt'), ate)
    # noinspection PyTypeChecker
    np.savetxt(pjoin(experiment_path, 'rpe_r.txt'), error_r)
    # noinspection PyTypeChecker
    np.savetxt(pjoin(experiment_path, 'rpe_t.txt'), error_t)

    def rmse(x: np.ndarray) -> float:
        return np.sqrt(np.mean(np.square(x)))

    logging.info(f"{dataset_name} - {pred_label.upper()} vs. {gt_label.upper()}:")
    logging.info(f"\tATE: {rmse(ate):.2f} m")
    logging.info(f"\tRPE (rot): {rmse(np.rad2deg(error_r)):.2f}\N{DEGREE SIGN}")
    logging.info(f"\tRPE (tra): {rmse(error_t):.2f} m")

    set_key_path(results_dict, [dataset_name, pred_label, 'ate'], rmse(ate))
    set_key_path(results_dict, [dataset_name, pred_label, 'rpe', 'rotation'], rmse(np.rad2deg(error_r)))
    set_key_path(results_dict, [dataset_name, pred_label, 'rpe', 'translation'], rmse(error_t))

    with temporary_trajectory(dataset, pred_trajectory):
        mesh = tsdf_fusion(dataset, background_mesh_options, frame_set=frame_set)
        mesh.export(pjoin(experiment_path, f"mesh.ply"))


def tsdf_fusion_with_colmap(dataset: HiveDataset, frame_set: List[int], mesh_options: BackgroundMeshOptions) \
        -> Optional[trimesh.Trimesh]:
    depth_folder = pjoin(dataset.base_path, 'colmap_depth')

    if not os.path.isdir(depth_folder):
        return None

    rgb_files = [dataset.rgb_dataset.image_filenames[i] for i in frame_set]
    mask_files = [dataset.mask_dataset.image_filenames[i] for i in frame_set]
    depth_files = [filename for filename in sorted(os.listdir(depth_folder))]
    poses = dataset.camera_trajectory[frame_set]
    camera_matrix = dataset.camera_matrix.copy()
    metadata = DatasetMetadata.from_json(dataset.metadata.to_json())
    # use length of depth files instead of frame set since depth files may be missing the last file.
    metadata.num_frames = len(depth_files)

    tmp_dir = 'tmp'

    if os.path.isdir(tmp_dir):
        shutil.rmtree(tmp_dir)

    os.makedirs(tmp_dir)

    metadata.save(pjoin(tmp_dir, HiveDataset.metadata_filename))
    np.savetxt(pjoin(tmp_dir, HiveDataset.camera_matrix_filename), camera_matrix)
    np.savetxt(pjoin(tmp_dir, HiveDataset.camera_trajectory_filename), poses)

    rgb_path = pjoin(tmp_dir, HiveDataset.rgb_folder)
    os.makedirs(rgb_path)

    def copy_rgb(index_filename):
        index, filename = index_filename
        src = pjoin(dataset.path_to_rgb_frames, filename)
        dst = pjoin(rgb_path, HiveDataset.index_to_filename(index))
        shutil.copy(src, dst)

    tqdm_imap(copy_rgb, list(enumerate(rgb_files)))

    def copy_depth(index_filename):
        index, filename = index_filename
        src = pjoin(depth_folder, filename)
        dst = pjoin(depth_path, HiveDataset.index_to_filename(index))
        shutil.copy(src, dst)

    depth_path = pjoin(tmp_dir, HiveDataset.depth_folder)
    os.makedirs(depth_path)
    tqdm_imap(copy_depth, list(enumerate(depth_files)))

    def copy_mask(index_filename):
        index, filename = index_filename
        src = pjoin(dataset.path_to_masks, filename)
        dst = pjoin(mask_path, HiveDataset.index_to_filename(index))
        shutil.copy(src, dst)

    mask_path = pjoin(tmp_dir, HiveDataset.mask_folder)
    os.makedirs(mask_path)
    tqdm_imap(copy_mask, list(enumerate(mask_files)))

    tmp_dataset = HiveDataset(tmp_dir)
    try:
        mesh = tsdf_fusion(tmp_dataset, mesh_options)
    except ValueError:  # ValueError is raised from the marching cubes function when tsdf_volume.min() > 0.
        mesh = None

    shutil.rmtree(tmp_dir)

    return mesh


@contextmanager
def virtual_display():
    display = os.environ['DISPLAY'] or ':99'
    width_height_depth = os.environ['XVFB_WHD'] or '1920x1080x24'

    cmd = shlex.split(f"Xvfb {display} -screen 0 {width_height_depth}")
    virtual_display_process = subprocess.Popen(cmd)

    try:
        yield
    finally:
        virtual_display_process.terminate()


@contextmanager
def temporary_camera_matrix(dataset: HiveDataset, camera_matrix: np.ndarray):
    """
    Context manager that temporarily replaces the camera matrix of a dataset.

    :param dataset: The dataset.
    :param camera_matrix: The camera matrix to use.
    """
    camera_matrix_backup = dataset.camera_matrix.copy()

    try:
        dataset.camera_matrix = camera_matrix

        yield
    finally:
        dataset.camera_matrix = camera_matrix_backup


@contextmanager
def disable_inpainted_data(dataset: HiveDataset):
    """
    Context manager that temporarily disables the inpainted data of a dataset.

    :param dataset: The dataset.
    """
    inpainted_rgb_dataset_backup = dataset.inpainted_rgb_dataset
    inpainted_depth_dataset_backup = dataset.inpainted_depth_dataset

    try:
        dataset.inpainted_rgb_dataset = None
        dataset.inpainted_depth_dataset = None

        yield
    finally:
        dataset.inpainted_rgb_dataset = inpainted_rgb_dataset_backup
        dataset.inpainted_depth_dataset = inpainted_depth_dataset_backup


class Latex:

    @staticmethod
    def to_mean_stddev(numbers: List[Union[int, float]],
                       formatter: Callable[[Union[float, int]], str] = '{:.2f}'.format):
        mean = statistics.mean(numbers)
        stddev = statistics.stdev(numbers)

        return f"{formatter(mean)} $\pm$ {formatter(stddev)}"

    @staticmethod
    def to_mean(numbers: List[Union[int, float]],
                formatter: Callable[[Union[float, int]], str] = '{:.2f}'.format):
        mean = statistics.mean(numbers)

        return formatter(mean)

    @staticmethod
    def format_key_for_latex(key: str) -> str:
        key_parts = key.split('_')
        key_parts = map(lambda string: string.capitalize(), key_parts)
        formatted_key = ' '.join(list(key_parts))

        return formatted_key

    @staticmethod
    def format_dataset_name(dataset_name: str) -> str:
        return dataset_name.replace('_', ' ')

    @staticmethod
    def format_timedelta(seconds) -> str:
        minutes, seconds = divmod(seconds, 60)

        return f"{minutes:02.0f}:{seconds:02.0f}"

    @staticmethod
    def format_int(number) -> str:
        return f"{number:,.0f}"

    @staticmethod
    def format_one_dp(number) -> str:
        return f"{number:,.1f}"

    @staticmethod
    def percent_formatter(number) -> str:
        return f"{100 * number:.2f}\\%"

    @staticmethod
    def sec_to_ms(number) -> str:
        return f"{number * 1e3:,.1f}"

    @staticmethod
    def bytes_to_megabytes(number) -> str:
        return f"{number * 1e-6:,.1f}"

    @staticmethod
    def bytes_to_gigabytes(number) -> str:
        return f"{number * 1e-9:,.1f}"


@dataclass
class MeshCompressionExperimentConfig:
    uncompressed_mesh_folder = 'mesh_uncompressed'
    compressed_mesh_folder = 'mesh_compressed'
    uncompressed_mesh_extension = '.ply'
    compressed_mesh_extension = '.drc'
    fg_mesh_name = 'fg'
    bg_mesh_name = 'bg'


class InpaintingExperiment:
    """Code for experiments where inpainted frame data are compared against the original frame data."""

    @classmethod
    def get_crop_regions(cls, rgb_frame, binary_mask, subdivisions=8):
        height, width = rgb_frame.shape[:2]
        segment_height = height // subdivisions
        segment_width = width // subdivisions

        for col in range(1, subdivisions - 1):
            for row in range(1, subdivisions - 1):
                col_start = col * segment_width
                col_end = col_start + segment_width
                row_start = row * segment_height
                row_end = row_start + segment_height

                region_mask = np.zeros((height, width), dtype=bool)
                region_mask[row_start:row_end, col_start:col_end] = True

                if np.any(region_mask & binary_mask):
                    logging.debug(f"Skipping region {col}-{row} due to overlap with segmentation mask...")
                    continue

                yield region_mask

    @classmethod
    def load_inpainting_model(cls):
        lama_weights_path = pjoin(os.environ["WEIGHTS_PATH"], 'big-lama')
        device = torch.device('cuda')

        train_config_path = os.path.join(lama_weights_path, 'config.yaml')
        with open(train_config_path, 'r') as f:
            train_config = OmegaConf.create(yaml.safe_load(f))

        train_config.training_model.predict_only = True
        train_config.visualizer.kind = 'noop'

        checkpoint_path = os.path.join(lama_weights_path,
                                       'models',
                                       'best.ckpt')
        model = load_checkpoint(train_config, checkpoint_path, strict=False, map_location='cpu')
        model.freeze()
        model.to(device)

        return model

    @classmethod
    def inpaint_rgb(cls, model, image, mask, pad_out_to_modulo=8):
        # Lama (and in general, PyTorch) expect image data to be in NCHW format. Here we first convert to CHW.
        image = np.transpose(image, (2, 0, 1))
        mask = np.broadcast_to(mask, (1, *mask.shape))
        # Lama, in particular, requires images to be normalized to the interval [0, 1] and stored as floats.
        image = (image / np.iinfo(image.dtype).max).astype(np.float32)
        mask = mask.astype(np.float32)

        frame_data = dict()
        frame_data['unpad_to_size'] = image.shape[1:]
        frame_data['image'] = pad_img_to_modulo(image, pad_out_to_modulo)
        frame_data['mask'] = pad_img_to_modulo(mask, pad_out_to_modulo)

        batch = default_collate([frame_data])

        with torch.no_grad():
            batch = move_to_device(batch, model.device)
            batch = model(batch)
            inpainted_frame = batch['inpainted'][0].permute(1, 2, 0).detach().cpu().numpy()
            unpad_to_size = batch.get('unpad_to_size', None)

            if unpad_to_size is not None:
                orig_height, orig_width = unpad_to_size
                inpainted_frame = inpainted_frame[:orig_height, :orig_width]

        inpainted_frame = np.clip(inpainted_frame * 255, 0, 255).astype('uint8')

        return inpainted_frame

    @classmethod
    def inpaint_depth(cls, depth_map, mask, depth_scale):
        depth_map_uint16 = (depth_map / depth_scale).astype(np.uint16)
        inpainted_depth = cv2.inpaint(depth_map_uint16, mask.astype(np.uint8), 30, cv2.INPAINT_TELEA)
        inpainted_depth[inpainted_depth < 0.0] = 0.0

        return inpainted_depth * depth_scale

    @classmethod
    def compare_rgb(cls, ref_image, est_image, mask, lpips_fn=None):
        v, u = mask.nonzero()
        crop_region_height = v.max() - v.min() + 1
        crop_region_width = u.max() - u.min() + 1
        rgb_raw_img_comp = ref_image[mask].reshape((crop_region_height, crop_region_width, 3))
        rgb_inpainted_img_comp = est_image[mask].reshape((crop_region_height, crop_region_width, 3))

        return compare_images(rgb_raw_img_comp, rgb_inpainted_img_comp, lpips_fn=lpips_fn)

    @classmethod
    def compare_depth(cls, ref_depth, est_depth, mask):
        valid_pixels_mask = (ref_depth > 0) & (est_depth > 0) & mask
        masked_ref_depth = ref_depth[valid_pixels_mask]
        masked_est_depth = est_depth[valid_pixels_mask]

        residuals = masked_ref_depth - masked_est_depth
        rmse = np.sqrt(np.mean(np.square(residuals)))
        abs_rel = np.mean(np.abs(residuals) / masked_ref_depth)
        max_ratio = np.maximum(masked_ref_depth / masked_est_depth, masked_est_depth / masked_ref_depth)
        delta_1 = np.mean(max_ratio <= 1.25)

        return {
            'rmse': float(rmse),
            'abs_rel': float(abs_rel),
            'delta_1': float(delta_1)
        }

    @classmethod
    def depth_to_img(cls, depth_map):
        depth_map_normalized = depth_map / depth_map.max()

        return (255 * depth_map_normalized).astype(np.uint8)

    @classmethod
    def mask_to_img(cls, mask):
        # Assume that the mask contains zeros and ones.
        return mask.astype(np.uint8) * 255


class LLFFAdaptor(VideoAdaptorBase):
    """
    Datasets in the multi-camera video formats of Neural 3D Video Synthesis from Multi-View Video
    (https://github.com/facebookresearch/Neural_3D_Video), and in particular the pose format of
    github.com/Fyusion/LLFF.
    """

    pose_filename = "poses_bounds.npy"
    required_files = [pose_filename]
    required_folders = []

    def __init__(self, base_path: File, output_path: File, num_frames=-1, frame_step=1, colmap_options=COLMAPOptions(),
                 resize_to: Optional[Union[int, Size]] = None, camera_feed: int = 0,
                 overwrite_colmap_data: bool = True):
        """
        :param base_path: The path to the dataset.
        :param output_path: The path to write the new dataset to.
        :param num_frames: The maximum of frames to process. Set to -1 (default) to process all frames.
        :param frame_step: The frequency to sample frames at for COLMAP and pose optimisation.
            If set to 1, samples all frames (i.e. no effect). Otherwise, if set to n > 1, samples every n frames.
        :param colmap_options: The configuration to use for COLMAP if estimating camera parameters.
        :param resize_to: The resolution (height, width) to resize the images to.
            If an int is given, the longest side will be scaled to this value and the shorter side will have its new
            length automatically calculated.
        :param camera_feed: Which camera feed to use.
        :param overwrite_colmap_data: Whether cached multicam COLMAP data should be discarded if it already exists.
        """
        contents = os.listdir(base_path)
        self.video_filenames = list(filter(lambda filename: Path(filename).suffix == '.mp4', contents))

        if len(self.video_filenames) < 1:
            raise FileNotFoundError(f"Dataset should have at least one video file, but found zero videos.")

        self.video_indices = [int(filename[3:5]) for filename in self.video_filenames]
        self.video_index_to_zero_index = {video_index: i for i, video_index in enumerate(self.video_indices)}

        if camera_feed not in self.video_indices:
            raise ValueError(f"Cannot use camera feed #{camera_feed}, valid camera feeds: {self.video_indices}.")

        self.camera_feed = camera_feed

        video_path = pjoin(base_path, self.video_filenames[self.camera_feed])

        super().__init__(base_path=base_path, output_path=output_path, num_frames=num_frames, frame_step=frame_step,
                         colmap_options=colmap_options, video_path=video_path, resize_to=resize_to)

        self.camera_matrix, self.pose_data_by_camera = self._get_scaled_colmap_pose_data(
            overwrite_ok=overwrite_colmap_data)

    def get_camera_matrix(self, camera_feed: Optional[int] = None) -> np.ndarray:
        return self.camera_matrix.matrix

    def get_pose(self, index: int, camera_feed: Optional[int] = None) -> np.ndarray:
        if camera_feed is None:
            camera_feed = self.camera_feed

        return self.pose_data_by_camera[camera_feed]

    def get_camera_trajectory(self, camera_feed: Optional[int] = None) -> Trajectory:
        if camera_feed is None:
            camera_feed = self.camera_feed

        pose = self.get_pose(0, camera_feed=camera_feed)
        num_frames = self.num_frames
        pose_data = np.repeat(pose.reshape((1, -1)), num_frames, axis=0)

        return Trajectory(pose_data)

    def get_frame(self, index: int, camera_feed: Optional[int] = None) -> np.ndarray:
        if camera_feed is None:
            camera_feed = self.camera_feed

        with self.open_video(os.path.join(self.base_path,
                                          self.video_filenames[self.video_index_to_zero_index[camera_feed]])) as video:
            video.set(cv2.CAP_PROP_POS_FRAMES, index)
            has_frame, frame = video.read()

            if not has_frame:
                raise IndexError(f"No frame data available for index {index:d}.")

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, dsize=(self.target_width, self.target_height), interpolation=cv2.INTER_CUBIC)

        return frame

    def _get_scaled_colmap_camera_params(self, colmap_processor: COLMAPProcessor,
                                         output_depth_folder: str,
                                         metadata: DatasetMetadata,
                                         frames_subset: List[int]) -> Tuple[np.ndarray, Trajectory]:
        return self.camera_matrix.matrix, self.get_camera_trajectory(self.camera_feed)

    def _get_scaled_colmap_pose_data(self, overwrite_ok: bool) -> Tuple[CameraMatrix, Dict[int, np.ndarray]]:
        """
        Run COLMAP and scale the pose data with estimated depth maps.

        LLFF datasets use COLMAP pose data, so we cannot use them directly in HIVE since HIVE assumes metric scale
        and the COLMAP poses are subject to an unknown scale.
        The LLFF datasets do not include the 3D point data, so we are unable to calculate the scaling factor directly.
        Therefore, we must run COLMAP ourselves.

        :param overwrite_ok: Whether to overwrite existing cached data (COLMAP, estimated depth maps).
        :return: The scaled camera parameters (intrinsic and extrinsic).
        """
        logging.info(f"Estimating camera parameters from multiple cameras with COLMAP...")
        output_folder = os.path.join(self.output_path, 'multicam_colmap')
        per_camera_frames_path = os.path.join(output_folder, 'frames')
        per_camera_estimated_depth_path = os.path.join(output_folder, 'depth')
        os.makedirs(per_camera_estimated_depth_path, exist_ok=True)
        os.makedirs(per_camera_frames_path, exist_ok=True)
        logging.debug(f"Extracting first frame from videos to {per_camera_frames_path}...")

        if overwrite_ok or len(os.listdir(per_camera_frames_path)) != len(self.video_filenames):
            for camera_feed in self.video_indices:
                frame = self.get_frame(index=0, camera_feed=camera_feed)
                frame_filename = HiveDataset.index_to_filename(camera_feed, '.jpg')
                Image.fromarray(frame).save(os.path.join(per_camera_frames_path, frame_filename))

        cm = COLMAPProcessor(per_camera_frames_path,
                             os.path.join(output_folder, 'workspace'),
                             self.colmap_options)

        if not cm.probably_has_results or overwrite_ok:
            cm.run(use_masks=False)

        cm_intrinsic, cm_extrinsic = cm.load_camera_params()
        camera_matrix = CameraMatrix.from_matrix(cm_intrinsic,
                                                 size=(self.target_height, self.target_width))
        cm_depth_maps = cm.get_sparse_depth_maps(cm_intrinsic, cm_extrinsic)
        # COLMAPProcessor interpolates poses for indices without a corresponding image (in our case, camera feeds that
        # were removed due to being out of sync). We do not want poses or depth maps for non-existent camera feeds,
        # so we filter them out.
        cm_extrinsic = Trajectory(cm_extrinsic[self.video_indices])
        cm_depth_maps = cm_depth_maps[self.video_indices]

        if (not os.path.isdir(per_camera_estimated_depth_path) or
            not len(os.listdir(per_camera_estimated_depth_path)) == len(cm_depth_maps)) \
                or overwrite_ok:
            logging.debug(f"Estimating depth maps for initial camera frames...")
            shutil.rmtree(per_camera_estimated_depth_path)
            estimate_depth_dpt(ImageFolderDataset(per_camera_frames_path), output_path=per_camera_estimated_depth_path)

        def depth_transform(depth_map):
            depth_map = depth_map.astype(np.float32) * HiveDataset.depth_scaling_factor
            depth_map[depth_map > 10.] = 0.0

            return depth_map

        est_depth_dataset = ImageFolderDataset(per_camera_estimated_depth_path, transform=depth_transform)

        est_depth = np.array(est_depth_dataset)
        nonzero_mask = (cm_depth_maps > 0.) & (est_depth > 0.)
        scaling_factor = np.median(est_depth[nonzero_mask] / cm_depth_maps[nonzero_mask])

        scaled_poses = cm_extrinsic.scale_trajectory(scaling_factor)
        scaled_poses = scaled_poses.normalise()

        pose_map = dict()

        for zero_index, camera_feed in enumerate(self.video_indices):
            pose_map[camera_feed] = scaled_poses[zero_index]

        return camera_matrix, pose_map


class LLFFExperiment:
    """
    Dataset from Li, Tianye, Mira Slavcheva, Michael Zollhoefer, Simon Green, Christoph Lassner, Changil Kim,
    Tanner Schmidt et al. "Neural 3d video synthesis from multi-view video." In Proceedings of the IEEE/CVF Conference
    on Computer Vision and Pattern Recognition, pp. 5521-5531. 2022.

    Uses dataset format of Mildenhall, Ben, Pratul P. Srinivasan, Rodrigo Ortiz-Cayon, Nima Khademi Kalantari,
    Ravi Ramamoorthi, Ren Ng, and Abhishek Kar. "Local light field fusion: Practical view synthesis with prescriptive
    sampling guidelines." ACM Transactions on Graphics (TOG) 38, no. 4 (2019): 1-14.
    """
    dataset_results_filename = 'metrics.json'
    url_format = "https://github.com/facebookresearch/Neural_3D_Video/releases/download/v1.0/{}"
    sequence_names = [
        'coffee_martini',
        'cook_spinach',
        'cut_roasted_beef',
        'flame_steak',
        'sear_steak',
        'flame_salmon_1'
    ]

    @classmethod
    def fetch_dataset(cls, data_folder: str, dataset_name: str):
        dataset_path = os.path.join(data_folder, dataset_name)

        if not os.path.isdir(dataset_path):
            try:
                logging.info(f"Downloading {dataset_name}...")
                zip_filename = f"{dataset_name}.zip"
                zip_path = os.path.join(data_folder, zip_filename)

                if dataset_name == "flame_salmon_1" and not os.path.isfile(zip_path):
                    zip_part_filenames = [
                        "flame_salmon_1_split.z01",
                        "flame_salmon_1_split.z02",
                        "flame_salmon_1_split.z03",
                        "flame_salmon_1_split.zip",
                    ]
                    logging.debug(f"Downloading zip parts {zip_part_filenames}...")

                    for zip_part_filename in zip_part_filenames:
                        zip_part_path = os.path.join(data_folder, zip_part_filename)

                        if os.path.isfile(zip_part_path):
                            continue

                        url = cls.url_format.format(zip_part_filename)
                        logging.debug(f"Downloading zip part from {url}...")
                        urllib.request.urlretrieve(url, zip_part_path)

                    logging.debug(f"Joining zip parts...")
                    command = f"zip -F {os.path.join(data_folder, 'flame_salmon_1_split.zip')} --out {zip_path}"
                    subprocess.run(shlex.split(command))

                    logging.debug(f"Deleting zip parts...")
                    for zip_part_filename in zip_part_filenames:
                        os.remove(os.path.join(data_folder, zip_part_filename))
                elif not os.path.isfile(zip_path):
                    url = cls.url_format.format(zip_filename)
                    urllib.request.urlretrieve(url, zip_path)

                logging.debug(f"Extracting zip file...")

                with zipfile.ZipFile(zip_path, 'r') as file:
                    file.extractall(data_folder)

                logging.info(f"Downloaded dataset {dataset_name} and extracted to {dataset_path}.")
            except Exception:
                raise FileNotFoundError(f"Could not find dataset at {dataset_path} or automatically download it.")

    @dataclass
    class Config:
        name: str
        camera_matrix: CameraMatrix
        fg_mesh: trimesh.Trimesh
        bg_mesh: trimesh.Trimesh

    @dataclass
    class RenderResult:
        dataset: str
        camera_feed: int
        config: str
        ssim: float
        psnr: float
        lpips: float
        ssim_masked: float
        psnr_masked: float
        lpips_masked: float

        def to_json(self) -> dict:
            return self.__dict__

        def to_structured_np_array(self):
            return np.array(
                [(self.dataset, self.camera_feed, self.config,
                  self.ssim, self.psnr, self.lpips,
                  self.ssim_masked, self.psnr_masked, self.lpips_masked)],
                dtype=[('dataset', 'U16'), ('camera_feed', int), ('config', 'U16'),
                       ('ssim', float), ('psnr', float), ('lpips', float),
                       ('ssim_masked', float), ('psnr_masked', float), ('lpips_masked', float)]
            )

    @classmethod
    def _get_multicam_config(cls, dataset_adaptor: LLFFAdaptor, dataset: HiveDataset, pipeline: Pipeline,
                             frame_index: int) -> Config:
        camera_matrix = dataset_adaptor.camera_matrix

        fg_mesh = pipeline.process_frame(dataset, index=frame_index)
        bg_mesh = pipeline.create_static_mesh(dataset)

        return cls.Config("multicam", camera_matrix, fg_mesh, bg_mesh)

    @classmethod
    def _get_kinect_config(cls, dataset: HiveDataset, pipeline: Pipeline, frame_index: int) -> Config:
        kinect_camera_matrix = KinectSensor.get_camera_matrix()

        with temporary_camera_matrix(dataset, kinect_camera_matrix.matrix):
            fg_mesh_kinect = pipeline.process_frame(dataset, index=frame_index)
            bg_mesh_kinect = pipeline.create_static_mesh(dataset)

        return cls.Config("monocular", kinect_camera_matrix, fg_mesh_kinect, bg_mesh_kinect)

    @classmethod
    def _get_cc_analysis_config(cls, dataset: HiveDataset, pipeline: Pipeline, frame_index: int) -> Config:
        kinect_camera_matrix = KinectSensor.get_camera_matrix()

        with temporary_camera_matrix(dataset, kinect_camera_matrix.matrix):
            fg_mesh_kinect = pipeline.process_frame(dataset, index=frame_index, enable_cc_analysis=False)
            bg_mesh_kinect = pipeline.create_static_mesh(dataset)

        return cls.Config("no_cc_analysis", kinect_camera_matrix, fg_mesh_kinect, bg_mesh_kinect)

    @classmethod
    def _get_bundlefusion_config(cls, dataset: HiveDataset, pipeline: Pipeline, frame_index: int,
                                 mesh_options=BackgroundMeshOptions(), num_frames=-1) -> Config:
        kinect_camera_matrix = KinectSensor.get_camera_matrix()

        with (temporary_camera_matrix(dataset, kinect_camera_matrix.matrix)):
            fg_mesh = pipeline.process_frame(dataset, index=frame_index)

            try:
                bf_mesh = bundle_fusion(output_folder='bundle_fusion', dataset=dataset, options=mesh_options,
                                        num_frames=num_frames)
            except RuntimeError as e:
                logging.warning(f"Encountered error while running BundleFusion: {e}\nUsing empty mesh.")
                bf_mesh = trimesh.Trimesh()

        return cls.Config("bundlefusion", kinect_camera_matrix, fg_mesh, bf_mesh)

    @classmethod
    def _get_no_inpainting_config(cls, dataset: HiveDataset, pipeline: Pipeline, frame_index: int) -> Config:
        kinect_camera_matrix = KinectSensor.get_camera_matrix()

        with disable_inpainted_data(dataset), temporary_camera_matrix(dataset, kinect_camera_matrix.matrix):
            fg_mesh_no_inpainted = pipeline.process_frame(dataset, index=frame_index)
            bg_mesh_no_inpainted = pipeline.create_static_mesh(dataset)

        return cls.Config("no_inpainting", kinect_camera_matrix, fg_mesh_no_inpainted, bg_mesh_no_inpainted)

    @classmethod
    def _get_compression_config(cls, output_folder: str, kinect_config: Config) -> Config:
        def save_draco(mesh: trimesh.Trimesh, path: str) -> str:
            # noinspection PyUnresolvedReferences
            if isinstance(mesh.visual, trimesh.visual.TextureVisuals):
                # Use vertex colours instead of textures since they seem to get lost for some reason?
                mesh.visual = mesh.visual.to_color()

            with open(path, 'wb') as f:
                # noinspection PyUnresolvedReferences
                f.write(trimesh.exchange.ply.export_draco(mesh))

            return path

        def load_draco(path: str) -> trimesh.Trimesh:
            with open(path, 'rb') as f:
                # noinspection PyUnresolvedReferences
                mesh_data = trimesh.exchange.ply.load_draco(f)

            return trimesh.Trimesh(**mesh_data)

        temp_mesh_path = os.path.join(output_folder, 'temp.drc')

        fg_mesh_draco = load_draco(save_draco(kinect_config.fg_mesh, temp_mesh_path))
        bg_mesh_draco = load_draco(save_draco(kinect_config.bg_mesh, temp_mesh_path))

        os.remove(temp_mesh_path)

        return cls.Config("compression", kinect_config.camera_matrix, fg_mesh_draco, bg_mesh_draco)

    @classmethod
    def compare_renders(cls, data_folder: str, sequence_name: str, output_folder: str, results_folder: str,
                        frame_index: int, lpips_fn: LPIPS,
                        no_cache: bool = False):
        logging.info(f"Running experiment for {sequence_name}...")

        dataset_path = os.path.join(data_folder, sequence_name)
        converted_dataset_path = os.path.join(output_folder, sequence_name)

        cls.fetch_dataset(data_folder=data_folder, dataset_name=sequence_name)

        logging.debug(f"Creating mesh data...")
        # TODO: Get config from `Experiments` object.
        frame_width = 640
        frame_height = 480
        num_frames = 300
        frame_step = 15

        dataset_adaptor = LLFFAdaptor(base_path=dataset_path,
                                      output_path=converted_dataset_path,
                                      num_frames=num_frames,
                                      frame_step=frame_step,
                                      colmap_options=COLMAPOptions(quality='medium'),
                                      resize_to=(frame_height, frame_width),
                                      camera_feed=0,
                                      overwrite_colmap_data=no_cache)
        dataset = dataset_adaptor.convert(estimate_pose=False,
                                          estimate_depth=True,
                                          inpainting_mode=InpaintingMode.Lama_Image_CV2_Depth,
                                          static_camera=False,
                                          no_cache=no_cache)
        pipeline = Pipeline(options=PipelineOptions(num_frames=num_frames, frame_step=frame_step),
                            storage_options=StorageOptions(dataset_path=dataset_path,
                                                           output_path=converted_dataset_path))

        kinect_config = cls._get_kinect_config(dataset, pipeline, frame_index)

        configurations = (
            cls._get_multicam_config(dataset_adaptor, dataset, pipeline, frame_index),
            kinect_config,
            cls._get_bundlefusion_config(dataset, pipeline, frame_index),
            cls._get_cc_analysis_config(dataset, pipeline, frame_index),
            cls._get_no_inpainting_config(dataset, pipeline, frame_index),
            cls._get_compression_config(output_folder, kinect_config),
        )

        logging.debug(f"Gathering frame data and camera parameters...")
        results = []

        for camera_feed in dataset_adaptor.video_indices:
            label = f"cam{camera_feed:02d}"
            logging.info(f"Running comparison for {label} in {results_folder}...")

            frame = dataset_adaptor.get_frame(index=frame_index, camera_feed=camera_feed)

            frame_path = os.path.join(results_folder, f"{label}.jpg")
            Image.fromarray(frame).save(frame_path)
            logging.debug(f"Wrote frame data to {frame_path}.")

            for config in configurations:
                logging.info(f"{config.name} config...")
                color = cls.render_mesh(config.camera_matrix,
                                        dataset_adaptor.get_pose(index=frame_index, camera_feed=camera_feed),
                                        config.fg_mesh, config.bg_mesh)
                color_np = np.asarray(color)

                screen_capture_path = os.path.join(results_folder, f"{label}_{config.name}.jpg")
                color.save(screen_capture_path)
                logging.debug(f"Wrote screen capture (color) to {screen_capture_path}.")

                ssim, psnr, lpips = compare_images(frame, color_np, lpips_fn=lpips_fn)

                masked_frame = frame.copy()
                masked_frame[color_np == [255, 255, 255]] = 255
                ssim_masked, psnr_masked, lpips_masked = compare_images(masked_frame, color_np, lpips_fn=lpips_fn)

                results.append(cls.RenderResult(sequence_name, camera_feed, config.name,
                                                ssim, psnr, lpips,
                                                ssim_masked, psnr_masked, lpips_masked))

        metrics_path = os.path.join(results_folder, cls.dataset_results_filename)

        with open(metrics_path, 'w') as f:
            json.dump([result.to_json() for result in results], f)
            logging.debug(f"Wrote metrics to {metrics_path}.")

    @classmethod
    def render_mesh(cls, camera_matrix: CameraMatrix, camera_pose: np.ndarray, *meshes: trimesh.Trimesh):
        # This rotation corrects for the rotation applied for viewing in the web-based renderer.
        # rotation = trimesh.transformations.rotation_matrix(angle=math.pi, direction=[1, 0, 0])
        rotation = np.diag([1., -1., -1., 1.])  # Converts from COLMAP [x -y, -z] to right-handed [x, y, z].
        pose_norm = rotation @ pose_vec2mat(camera_pose)

        camera = pyrender.PerspectiveCamera(yfov=camera_matrix.fov_y, aspectRatio=camera_matrix.aspect_ratio)
        scene = pyrender.Scene()

        for mesh in meshes:
            if mesh.is_empty:  # This may happen for the bundlefusion config if BundleFusion fails.
                continue

            scene.add(pyrender.Mesh.from_trimesh(mesh))

        scene.add(camera, pose=np.linalg.inv(pose_norm))
        renderer = pyrender.OffscreenRenderer(camera_matrix.width, camera_matrix.height)

        color, _ = renderer.render(scene, flags=pyrender.constants.RenderFlags.FLAT)
        color = Image.fromarray(color)
        renderer.delete()

        return color

    @classmethod
    def gather_results(cls, output_folder: str):
        dataset_names = list(filter(lambda item: os.path.isdir(os.path.join(output_folder, item)),
                                    os.listdir(output_folder)))

        results_list: List[np.ndarray] = []

        for dataset_name in dataset_names:
            dataset_path = os.path.join(output_folder, dataset_name)
            results_path = os.path.join(dataset_path, cls.dataset_results_filename)

            with open(results_path, 'r') as f:
                json_data = json.load(f)

            results_list += [cls.RenderResult(**json_row).to_structured_np_array() for json_row in json_data]

        import numpy.lib.recfunctions as rfn

        return rfn.stack_arrays(results_list)

    @classmethod
    def export_latex(cls, results_records: np.ndarray, summary_path: str, latex_path: str):
        df = pd.DataFrame.from_records(results_records)

        summary_path = os.path.join(summary_path, 'llff.json')
        logging.info(f"Exporting summary to {summary_path}...")
        df.to_json(summary_path, orient='records')

        df.columns = df.columns.str.replace('_', ' ')
        df['config'] = df['config'].str.replace('_', ' ')
        df = df.drop(['dataset'], axis='columns')
        df['camera feed'] = np.where(df['camera feed'] == 0, '00', '01-20')

        df_not_masked = df.drop(['ssim masked', 'psnr masked', 'lpips masked'], axis='columns')
        df_masked = df.drop(['ssim', 'psnr', 'lpips'], axis='columns')
        df_masked['camera feed'] = df_masked['camera feed'].str.replace('01-20', r'01-20$^\dagger$')
        df_masked.columns = df_masked.columns.str.replace(' masked', '')

        df_not_masked = df_not_masked.groupby(by=['config', 'camera feed']).mean()
        df_masked = df_masked.groupby(by=['config', 'camera feed']).mean()
        df_masked = df_masked.drop('00', level='camera feed')

        df_merged = pd.concat((df_not_masked, df_masked), axis='rows').sort_index(axis=0)
        config_order = ['multicam', 'monocular', 'bundlefusion', 'compression', 'no cc analysis', 'no inpainting']
        camera_order = ['00', '01-20', r'01-20$^\dagger$']
        df_merged = df_merged.reindex(config_order, level='config').reindex(camera_order, level='camera feed')
        df_merged.index = df_merged.index.rename(['Config', 'Camera Feed'])
        df_merged.index = df_merged.index.set_levels(
            ['Multicam', 'Monocular', 'BundleFusion', 'Compression', 'No CC Analysis', 'No Inpainting'],
            level='Config'
        )
        df_merged = df_merged.rename(str.upper, axis='columns')

        latex_path = os.path.join(latex_path, 'llff.tex')

        logging.info(f"Exporting latex to {latex_path}...")
        df_merged.to_latex(latex_path, float_format="%.2f")


class DynamicScenesExperiments:
    """
    Dataset from Gao, Chen, Ayush Saraf, Johannes Kopf, and Jia-Bin Huang. "Dynamic view
    synthesis from dynamic monocular video." In Proceedings of the IEEE/CVF International Conference on Computer Vision,
    pp. 5712-5721. 2021. (https://github.com/gaochen315/DynamicNeRF).
    """
    url = "https://filebox.ece.vt.edu/~chengao/free-view-video/data.zip"
    zip_filename = "data.zip"

    sequence_names = [
        'Balloon1',
        'Balloon2',
        'Jumping',
        'Playground',
        'Skating',
        'Truck',
        'Umbrella',
    ]

    @classmethod
    def fetch_dataset(cls, data_folder: str, sequence_names: List[str]):
        for sequence_name in sequence_names:
            if not os.path.isdir(os.path.join(data_folder, sequence_name)):
                break
        else:
            return

        try:
            logging.info(f"Downloading Dynamic Scenes Dataset from {cls.url}...")
            zip_path = os.path.join(data_folder, cls.zip_filename)

            if not os.path.isfile(zip_path):
                urllib.request.urlretrieve(cls.url, zip_path)

            logging.debug(f"Extracting zip file...")

            with zipfile.ZipFile(zip_path, 'r') as archive:
                archive.extractall(data_folder)

            # data.zip has the first subdirectory 'data'
            extracted_path = os.path.join(data_folder, 'data')

            for sequence_name in os.listdir(extracted_path):
                shutil.move(src=os.path.join(extracted_path, sequence_name),
                            dst=os.path.join(data_folder, sequence_name))

            os.rmdir(extracted_path)

            logging.info(f"Downloaded dataset and extracted to {data_folder}.")
        except Exception:
            raise FileNotFoundError(f"Could not find dataset at {data_folder} or automatically download it.")


@dataclass(frozen=True)
class HyperNeRFCamera:
    orientation: np.ndarray  # 3x3 Rotation matrix
    position: np.ndarray  # 1x3 translation vector
    focal_length: float
    principal_point: np.ndarray  # cx, cy
    skew: float
    radial_distortion: np.ndarray
    tangential_distortion: np.ndarray
    image_size: np.ndarray  # width, height

    @property
    def width(self) -> int:
        return int(self.image_size[0])

    @property
    def height(self) -> int:
        return int(self.image_size[1])

    @property
    def cx(self) -> int:
        return int(self.principal_point[0])

    @property
    def cy(self) -> int:
        return int(self.principal_point[1])

    def to_camera_matrix(self) -> CameraMatrix:
        return CameraMatrix(
            fx=self.focal_length,
            fy=self.focal_length,
            cx=self.cx,
            cy=self.cy,
            width=self.width,
            height=self.height
        )

    def to_pose_matrix(self) -> np.ndarray:
        matrix = np.eye(4, dtype=float)

        matrix[:3, :3] = self.orientation
        matrix[:3, 3] = self.position

        return matrix

    @classmethod
    def from_json(cls, path: str) -> 'HyperNeRFCamera':
        with open(path, 'r') as f:
            json_data = json.load(f)

        return HyperNeRFCamera(
            orientation=np.asarray(json_data['orientation'], dtype=float),
            position=np.asarray(json_data['position'], dtype=float),
            focal_length=float(json_data['focal_length']),
            principal_point=np.asarray(json_data['principal_point'], dtype=float),
            skew=float(json_data['skew']),
            radial_distortion=np.asarray(json_data['radial_distortion'], dtype=float),
            tangential_distortion=np.asarray(json_data['tangential_distortion'], dtype=float),
            image_size=np.asarray(json_data['image_size'], dtype=int),
        )


@dataclass(frozen=True)
class HyperNeRFDatasetConfig:
    count: int
    num_exemplars: int
    ids: List[str]
    train_ids: List[str]
    val_ids: List[str]

    @classmethod
    def from_json(cls, path: str) -> 'HyperNeRFDatasetConfig':
        with open(path, 'r') as f:
            json_data = json.load(f)

        return HyperNeRFDatasetConfig(
            count=int(json_data['count']),
            num_exemplars=int(json_data['num_exemplars']),
            ids=list(json_data['ids']),
            train_ids=list(json_data['train_ids']),
            val_ids=list(json_data['val_ids'])
        )


@dataclass(frozen=True)
class HyperNeRFMetadata:
    time_id: int
    warp_id: int
    appearance_id: int
    camera_id: int

    @classmethod
    def map_from_json(cls, path: str) -> Dict[str, 'HyperNeRFMetadata']:
        with open(path, 'r') as f:
            json_data = json.load(f)

        return {
            image_id: HyperNeRFMetadata(
                time_id=int(metadata['time_id']),
                warp_id=int(metadata['warp_id']),
                appearance_id=int(metadata['appearance_id']),
                camera_id=int(metadata['camera_id'])
            )
            for image_id, metadata in json_data.items()
        }


@dataclass(frozen=True)
class HyperNeRFSceneConfig:
    scale: float
    scene_to_metric: float
    center: np.ndarray  # shape = (3,)
    near: float
    far: float

    @classmethod
    def from_json(cls, path: str) -> 'HyperNeRFSceneConfig':
        with open(path, 'r') as f:
            json_data = json.load(f)

        return HyperNeRFSceneConfig(
            scale=float(json_data['scale']),
            scene_to_metric=float(json_data['scene_to_metric']),
            center=np.asarray(json_data['center'], dtype=float),
            near=float(json_data['near']),
            far=float(json_data['far'])
        )


class HyperNeRFAdaptor(DatasetAdaptor):
    dataset_filename = 'dataset.json'
    metadata_filename = 'metadata.json'
    points_filename = 'points.npy'
    scene_filename = 'scene.json'

    required_files = [dataset_filename, metadata_filename, points_filename, scene_filename]

    camera_folder = 'camera'
    rgb_folder = 'rgb'

    camera_filename_format = "{camera_id:s}.json"

    scales = {1, 2, 4, 8, 16}
    rgb_sub_folder_format = "{scale:d}x"

    def __init__(self, base_path: File, output_path: File, num_frames=-1, frame_step=1, colmap_options=COLMAPOptions(),
                 scale: int = 2, is_train=True):
        """
        :param base_path: The root directory of the HyperNeRF dataset.
        :param output_path: Where to save the converted dataset.
        :param num_frames: The number of frames to process. Set to -1 to process all frames.
        :param frame_step: How many often to sample frames for COLMAP. Setting to 1 uses all frames, 30 uses 1 frame
            every 30 frames.
        :param colmap_options: The configuration to run COLMAP with.
        :param scale: The integer scale of the images to use (most papers use scale=2, i.e. half-resolution).
        :param is_train: Whether to use the training image set (`True`) or the validation image set (`False`).
        """
        assert scale in self.scales, f"Scale must be one of {self.scales}, but got {scale}."

        super().__init__(base_path=base_path, output_path=output_path, num_frames=num_frames, frame_step=frame_step,
                         colmap_options=colmap_options)

        self.dataset = HyperNeRFDatasetConfig.from_json(self._dataset_path)
        self.metadata = HyperNeRFMetadata.map_from_json(self._metadata_path)
        self.scene = HyperNeRFSceneConfig.from_json(self._scene_path)
        self.cameras = {camera_id: HyperNeRFCamera.from_json(self._camera_path(camera_id))
                        for camera_id in self.dataset.ids}

        self.scale = scale
        self.is_train = is_train

        if self.num_frames < 1:  # e.g., -1
            self.num_frames = len(self.dataset.train_ids)
        else:
            self.num_frames = min(len(self.dataset.train_ids), self.num_frames)

    @property
    def _dataset_path(self) -> str:
        return os.path.join(self.base_path, self.dataset_filename)

    @property
    def _metadata_path(self) -> str:
        return os.path.join(self.base_path, self.metadata_filename)

    @property
    def _points_path(self) -> str:
        return os.path.join(self.base_path, self.points_filename)

    @property
    def _scene_path(self) -> str:
        return os.path.join(self.base_path, self.scene_filename)

    def _camera_path(self, camera_id: str) -> str:
        return os.path.join(self.base_path, self.camera_folder, self.camera_filename_format.format(camera_id=camera_id))

    def get_metadata(self, estimate_pose: bool, estimate_depth: bool, scale: Optional[int] = None) -> DatasetMetadata:
        if scale is None:
            scale = self.scale

        camera = self.cameras[self.dataset.val_ids[0]]
        depth_mask_iterations = BackgroundMeshOptions().depth_mask_dilation_iterations

        return DatasetMetadata(
            self.num_frames, fps=30.0,  # fps is a rough estimate
            width=camera.width // scale, height=camera.height // scale,
            estimate_pose=estimate_pose,
            estimate_depth=estimate_depth,
            depth_mask_dilation_iterations=depth_mask_iterations,
            depth_scale=HiveDataset.depth_scaling_factor,
            frame_step=self.frame_step, colmap_options=self.colmap_options
        )

    def _validate_scale(self, scale: int):
        assert scale in self.scales, f"Scale must be one of {self.scales}, but got {scale}."

    def _get_frame_id(self, index: int, is_train: bool):
        return self.dataset.train_ids[index] if is_train else self.dataset.val_ids[index]

    def get_frame(self, index: int, scale: Optional[int] = None, is_train: Optional[bool] = None) -> np.ndarray:
        if scale is None:
            scale = self.scale

        if is_train is None:
            is_train = self.is_train

        self._validate_scale(scale)

        rgb_sub_folder = self.rgb_sub_folder_format.format(scale=scale)
        frame_name = self._get_frame_id(index, is_train)
        frame_path = os.path.join(self.base_path, self.rgb_folder, rgb_sub_folder, f"{frame_name}.png")

        return imageio.v3.imread(frame_path)

    def get_camera_matrix_wrapper(self, index=0, scale: Optional[int] = None, is_train: Optional[bool] = None):
        if scale is None:
            scale = self.scale

        if is_train is None:
            is_train = self.is_train

        frame_id = self._get_frame_id(index, is_train)
        camera_matrix = self.cameras[frame_id].to_camera_matrix()

        target_width = camera_matrix.width // scale
        target_height = camera_matrix.height // scale

        camera_matrix = camera_matrix.scale((target_height, target_width))

        return camera_matrix

    def get_camera_matrix(self, index=0, scale: Optional[int] = None, is_train: Optional[bool] = None) -> np.ndarray:
        return self.get_camera_matrix_wrapper(index=index, scale=scale, is_train=is_train).matrix

    def get_pose(self, index: int, is_train: Optional[bool] = None, metric_scale=True) -> np.ndarray:
        if is_train is None:
            is_train = self.is_train

        frame_id = self._get_frame_id(index, is_train)
        camera_matrix = self.cameras[frame_id].to_pose_matrix()

        if metric_scale:
            camera_matrix[:3, 3] *= self.scene.scene_to_metric

        return pose_mat2vec(camera_matrix)

    def get_camera_trajectory(self, is_train: Optional[bool] = None, metric_scale=True) -> Trajectory:
        if is_train is None:
            is_train = self.is_train

        trajectory = np.vstack([self.get_pose(i, is_train=is_train, metric_scale=metric_scale)
                                for i in range(self.num_frames)])
        return Trajectory(trajectory)


class HyperNeRFExperiments:
    """
    Uses dataset from Park, Keunhong, Utkarsh Sinha, Peter Hedman, Jonathan T. Barron, Sofien Bouaziz, Dan B. Goldman,
    Ricardo Martin-Brualla, and Steven M. Seitz. "Hypernerf: A higher-dimensional representation for topologically
    varying neural radiance fields." arXiv preprint arXiv:2106.13228 (2021).
    """
    url_format = "https://github.com/google/hypernerf/releases/download/v0.1/{}"

    sequence_names = [
        'vrig_3dprinter',
        'vrig_broom',
        'vrig_chicken',
        'vrig_peel-banana',
    ]

    sequence_to_folder = {sequence_name: sequence_name.replace('_', '-')
                          for sequence_name in sequence_names}

    @classmethod
    def fetch_sequence(cls, data_folder: str, sequence_name: str):
        dataset_path = os.path.join(data_folder, sequence_name)

        if not os.path.isdir(dataset_path):
            try:
                logging.info(f"Downloading {sequence_name}...")
                zip_filename = f"{sequence_name}.zip"
                zip_path = os.path.join(data_folder, zip_filename)

                if not os.path.isfile(zip_path):
                    url = cls.url_format.format(zip_filename)
                    urllib.request.urlretrieve(url, zip_path)

                logging.debug(f"Extracting zip file...")

                with zipfile.ZipFile(zip_path, 'r') as file:
                    file.extractall(data_folder)

                folder_name = cls.sequence_to_folder[sequence_name]
                extracted_path = os.path.join(data_folder, folder_name)

                if os.path.isdir(extracted_path):
                    shutil.move(src=extracted_path, dst=os.path.join(data_folder, sequence_name))

                logging.info(f"Downloaded dataset {sequence_name} and extracted to {dataset_path}.")
            except Exception:
                raise FileNotFoundError(f"Could not find dataset at {dataset_path} or automatically download it.")


# TODO: Pull out each experiment type into own class.
# TODO: Add experiments for view reconstruction from Novel View Synthesis datasets. Record cam00 (reference view)
#  metrics separately from the rest.
class Experiments:
    def __init__(self, data_path: str, output_path: str, overwrite_ok: bool, dataset_names: List[str],
                 num_frames: int, frame_step: int, log_file: str):
        self.data_path = data_path  # Where the datasets are stored.
        self.output_path = output_path  # Where to write all the results.
        self.overwrite_ok = overwrite_ok

        self.dataset_names = dataset_names

        for dataset_name in dataset_names:
            dataset_path = pjoin(data_path, dataset_name)

            if not os.path.isdir(dataset_path):
                raise FileNotFoundError(f"Could not find dataset at {dataset_path}")

        self.num_frames = num_frames
        self.frame_step = frame_step
        self.log_file = log_file

        self.mesh_compression_experiment_config = MeshCompressionExperimentConfig()

        self.gt_label = 'gt'
        self.cm_label = 'cm'
        self.dpt_label = 'dpt'
        self.est_label = 'est'

        self.labels = (self.gt_label, self.cm_label, self.est_label)

        self.inpainting_mode = InpaintingMode.Lama_Image_CV2_Depth

        self.gt_options = PipelineOptions(num_frames=num_frames, frame_step=frame_step,
                                          estimate_pose=False, estimate_depth=False,
                                          inpainting_mode=self.inpainting_mode, log_file=self.log_file)
        self.cm_options = PipelineOptions(num_frames=num_frames, frame_step=frame_step,
                                          estimate_pose=True, estimate_depth=False,
                                          inpainting_mode=self.inpainting_mode, log_file=self.log_file)
        self.dpt_options = PipelineOptions(num_frames=num_frames, frame_step=frame_step,
                                           estimate_pose=False, estimate_depth=True,
                                           inpainting_mode=self.inpainting_mode, log_file=self.log_file)
        self.est_options = PipelineOptions(num_frames=num_frames, frame_step=frame_step,
                                           estimate_pose=True, estimate_depth=True,
                                           inpainting_mode=self.inpainting_mode, log_file=self.log_file)

        self.mesh_options = BackgroundMeshOptions(key_frame_threshold=0.3)

        self.pipeline_configurations = [
            (self.gt_label, self.gt_options),
            (self.cm_label, self.cm_options),
            # (self.dpt_label, self.dpt_options),
            (self.est_label, self.est_options),
        ]

        self.dataset_paths = self.generate_dataset_paths(output_path, dataset_names, self.pipeline_configurations)

        self.summaries_path = pjoin(self.output_path, 'summaries')
        self.latex_path = pjoin(self.output_path, 'latex')
        self.trajectory_outputs_path = pjoin(self.output_path, 'trajectory')
        self.compression_outputs_path = pjoin(self.output_path, 'compression')
        self.inpainting_outputs_path = pjoin(self.output_path, 'inpainting')
        self.tmp_path = pjoin(self.output_path, 'tmp')

        for path in (self.summaries_path, self.latex_path, self.tmp_path, self.compression_outputs_path,
                     self.trajectory_outputs_path, self.inpainting_outputs_path):
            os.makedirs(path, exist_ok=True)

        self.pipeline_results_path = pjoin(self.summaries_path, 'pipeline.json')
        self.trajectory_results_path = pjoin(self.summaries_path, 'trajectory.json')
        self.kid_running_results_path = pjoin(self.summaries_path, 'kid_running.json')
        self.bundle_fusion_results_path = pjoin(self.summaries_path, 'bundle_fusion.json')
        self.compression_results_path = pjoin(self.summaries_path, 'compression.json')
        self.inpainting_results_path = pjoin(self.summaries_path, 'inpainting.json')

        self.colmap_options = COLMAPOptions(quality='medium')
        self.webxr_options = WebXROptions(webxr_path=self.tmp_path)

    @staticmethod
    def generate_dataset_paths(output_path: str,
                               dataset_names: List[str],
                               pipeline_configurations: List[Tuple[str, PipelineOptions]]) -> Dict[str, Dict[str, str]]:
        output_paths = dict()

        for dataset_name in dataset_names:
            output_paths[dataset_name] = dict()

            for label, config in pipeline_configurations:
                output_paths[dataset_name][label] = pjoin(output_path, f"{dataset_name}_{label}")

        return output_paths

    def run_kid_running_experiments(self, filename: str):
        """
        Run the experiments for the kid running sequence which include comparing inpainting modes and compression quality.

        :param filename: The filename of the kid running video (assumed to be located in the dataset folder).
        """
        dataset_path = pjoin(self.data_path, filename)

        if not os.path.isfile(dataset_path):
            raise FileExistsError(f"Dataset does not exist at: {dataset_path}")

        dataset_name = 'kid_running'
        no_compression_label = 'no_compression'

        configs = [
            (self.est_label, self.est_options),
            ('no_inpainting', PipelineOptions(num_frames=self.num_frames, frame_step=self.frame_step,
                                              estimate_pose=True, estimate_depth=True,
                                              inpainting_mode=InpaintingMode.Off, log_file=self.log_file)),
            ('cv_inpainting', PipelineOptions(num_frames=self.num_frames, frame_step=self.frame_step,
                                              estimate_pose=True, estimate_depth=True,
                                              inpainting_mode=InpaintingMode.CV2_Image_Depth, log_file=self.log_file)),
            (no_compression_label, self.est_options)
        ]

        kid_running_results = dict()

        for label, pipeline_options in configs:
            logging.info(f"Running pipeline for config (dataset, config): ({dataset_name}, {label})")

            output_path = pjoin(self.output_path, f"{dataset_name}_{label}")
            storage_options = StorageOptions(dataset_path=dataset_path,
                                             output_path=output_path,
                                             no_cache=True, overwrite_ok=True)
            profiling_json_path = pjoin(output_path, 'profiling.json')
            has_profiling_json = os.path.isfile(profiling_json_path)

            is_valid_dataset = HiveDataset.is_valid_folder_structure(output_path)

            if not is_valid_dataset or not has_profiling_json or self.overwrite_ok:
                pipeline = Pipeline(options=pipeline_options,
                                    storage_options=storage_options,
                                    colmap_options=self.colmap_options,
                                    webxr_options=self.webxr_options)
                pipeline.run(compress=label == no_compression_label)

            with open(profiling_json_path, 'r') as f:
                profile = json.load(f)
                set_key_path(kid_running_results, [dataset_name, label], profile)

        with open(self.kid_running_results_path, 'w') as f:
            json.dump(kid_running_results, f)

        logging.info(f"Saved kid running results to {self.kid_running_results_path}.")

    def run_pipeline_experiments(self):
        pipeline_results = dict()

        for dataset_name in self.dataset_names:
            pipeline_results[dataset_name] = dict()

            for label, config in self.pipeline_configurations:
                logging.info(f"Running pipeline for config (dataset, config): ({dataset_name}, {label})")
                storage_options = StorageOptions(dataset_path=pjoin(self.data_path, dataset_name),
                                                 output_path=self.dataset_paths[dataset_name][label],
                                                 no_cache=True, overwrite_ok=True)

                profiling_json_path = pjoin(self.dataset_paths[dataset_name][label], 'profiling.json')
                has_profiling_json = os.path.isfile(profiling_json_path)

                is_valid_dataset = HiveDataset.is_valid_folder_structure(self.dataset_paths[dataset_name][label])

                if not is_valid_dataset or not has_profiling_json or self.overwrite_ok:
                    pipeline = Pipeline(options=config,
                                        storage_options=storage_options,
                                        colmap_options=self.colmap_options,
                                        webxr_options=self.webxr_options)
                    pipeline.run()

                with open(profiling_json_path, 'r') as f:
                    profile = json.load(f)
                    pipeline_results[dataset_name][label] = profile

        with open(self.pipeline_results_path, 'w') as f:
            json.dump(pipeline_results, f)

        logging.info(f"Saved pipeline results to {self.pipeline_results_path}.")

    def export_pipeline_results(self):
        logging.info("Exporting pipeline results as LaTeX tables...")

        with open(self.pipeline_results_path, 'r') as f:
            pipeline_results = json.load(f)

        runtime_breakdown = dict()
        total_runtimes = []
        total_frame_times = []

        performance_statistics = {
            dataset_name: {
                label: {
                    'total_time': 0.0,
                    'ram_usage': 0.0,
                    'vram_usage': 0.0,
                }
                for label in self.labels
            }
            for dataset_name in self.dataset_names
        }

        file_statistics_by_layer = {
            layer: {
                'mesh_count': [],
                'time': [],
                'uncompressed_file_size': [],
                'compressed_file_size': [],
                'data_saving': [],
                'compression_ratio': [],
            }
            for layer in ('foreground', 'background')
        }

        # For each dataset and config:
        for dataset_name in self.dataset_names:
            for label in self.labels:
                stats = pipeline_results[dataset_name][label]
                frame_count = stats['frame_count']['total']

                # Runtime breakdown
                timing = stats['timing']

                total_runtimes.append(stats['elapsed_time']['total'])
                total_frame_times.append(stats['elapsed_time']['per_frame'])

                foreground_reconstruction = timing['foreground_reconstruction']
                foreground_wall_time_total = foreground_reconstruction['total']

                foreground_user_time_total = 0.0

                for sub_step in foreground_reconstruction:
                    if sub_step == 'total':
                        continue
                    elif sub_step == 'per_object_mesh':
                        foreground_user_time_total += foreground_reconstruction[sub_step]['total']['mean']
                    else:
                        foreground_user_time_total += foreground_reconstruction[sub_step]['mean']

                foreground_step_wall_times = dict()

                for sub_step in foreground_reconstruction:
                    if sub_step == 'total':
                        continue
                    elif sub_step == 'per_object_mesh':
                        ratio = foreground_reconstruction[sub_step]['total']['mean'] / foreground_user_time_total
                    else:
                        ratio = foreground_reconstruction[sub_step]['mean'] / foreground_user_time_total

                    foreground_step_wall_times[sub_step] = foreground_wall_time_total * ratio

                foreground_step_per_frame_wall_times = dict()

                for sub_step in foreground_reconstruction:
                    if sub_step == 'total':
                        continue
                    elif sub_step == 'per_object_mesh':
                        ratio = foreground_reconstruction[sub_step]['total']['mean'] / foreground_user_time_total
                        divisor = foreground_reconstruction[sub_step]['total']['count']
                    else:
                        ratio = foreground_reconstruction[sub_step]['mean'] / foreground_user_time_total
                        divisor = foreground_reconstruction[sub_step]['count']

                    foreground_step_per_frame_wall_times[sub_step] = (foreground_wall_time_total * ratio) / divisor

                if label == self.est_label:
                    def add_breakdown_stats(step, sub_step):
                        if step not in runtime_breakdown:
                            runtime_breakdown[step] = dict()

                        if sub_step not in runtime_breakdown[step]:
                            runtime_breakdown[step][sub_step] = {
                                'total_wall_time': [],
                                'frame_time': []
                            }

                        if sub_step == '-':
                            if isinstance(timing[step], dict):
                                time = timing[step]['total']
                            else:
                                time = timing[step]

                            runtime_breakdown[step][sub_step]['total_wall_time'].append(time)
                            runtime_breakdown[step][sub_step]['frame_time'].append(time / frame_count)
                        else:
                            runtime_breakdown[step][sub_step]['total_wall_time'].append(timing[step][sub_step])
                            runtime_breakdown[step][sub_step]['frame_time'].append(timing[step][sub_step] / frame_count)

                    def add_foreground_breakdown_stats(sub_step):
                        step = 'foreground_reconstruction'

                        if step not in runtime_breakdown:
                            runtime_breakdown[step] = dict()

                        if sub_step not in runtime_breakdown[step]:
                            runtime_breakdown[step][sub_step] = {
                                'total_wall_time': [],
                                'frame_time': []
                            }

                        runtime_breakdown[step][sub_step]['total_wall_time'].append(
                            foreground_step_wall_times[sub_step])
                        runtime_breakdown[step][sub_step]['frame_time'].append(
                            foreground_step_per_frame_wall_times[sub_step])

                    add_breakdown_stats('load_dataset', 'create_metadata')
                    add_breakdown_stats('load_dataset', 'copy_frames')
                    add_breakdown_stats('load_dataset', 'create_instance_segmentation_masks')
                    add_breakdown_stats('load_dataset', 'get_depth_maps')
                    add_breakdown_stats('load_dataset', 'get_camera_parameters')
                    add_breakdown_stats('load_dataset', 'inpainting')
                    add_breakdown_stats('background_reconstruction', '-')
                    add_foreground_breakdown_stats('binary_mask_creation')
                    add_foreground_breakdown_stats('per_object_mesh')
                    add_foreground_breakdown_stats('face_filtering')
                    add_foreground_breakdown_stats('mesh_decimation')
                    add_foreground_breakdown_stats('floater_removal')
                    add_foreground_breakdown_stats('texturing')
                    add_foreground_breakdown_stats('texture_atlas_packing')
                    add_breakdown_stats('scene_centering', '-')
                    add_breakdown_stats('mesh_export', '-')
                    add_breakdown_stats('mesh_compression', 'foreground')
                    add_breakdown_stats('mesh_compression', 'background')
                    add_breakdown_stats('webxr_export', '-')

                # Performance stats by dataset
                performance_statistics[dataset_name][label] = {
                    'total_time': sum([
                        timing['load_dataset']['total'],
                        timing['background_reconstruction']['total'],
                        timing['foreground_reconstruction']['total'],
                        timing['foreground_reconstruction']['total'],
                        timing['scene_centering'],
                        timing['mesh_export'],
                        timing['mesh_compression']['total'],
                        timing['webxr_export'],
                    ]),
                    'ram_usage': stats['peak_ram_usage'],
                    'vram_usage': stats['peak_vram_usage']['allocated'],
                }

                # Extract file size stats and mesh compression stats
                for layer in ('foreground', 'background'):
                    mesh_compression = stats['mesh_compression']

                    file_statistics_by_layer[layer]['mesh_count'].append(stats['frame_count'][layer])
                    file_statistics_by_layer[layer]['time'].append(stats['timing']['mesh_compression'][layer])
                    file_statistics_by_layer[layer]['uncompressed_file_size'].append(
                        mesh_compression[layer]['uncompressed_file_size'])
                    file_statistics_by_layer[layer]['compressed_file_size'].append(
                        mesh_compression[layer]['compressed_file_size'])
                    file_statistics_by_layer[layer]['data_saving'].append(mesh_compression[layer]['data_saving'])
                    file_statistics_by_layer[layer]['compression_ratio'].append(
                        mesh_compression[layer]['compression_ratio'])

        # Aggregate performance statistics

        # Runtime breakdown
        latex_lines = [
            r"\begin{tabular}{llrr}",
            r"\toprule",
            r"Step & Sub-Step & Total Time (mm:ss) & Frame Time (ms) \\"
        ]

        all_times = {'frame_time': [], 'total_wall_time': []}

        for step in runtime_breakdown:
            sub_step_count = len(runtime_breakdown[step])
            latex_lines.append(r"\midrule")
            latex_lines.append(r"\multirow{" + str(sub_step_count) + "}{*}{" + Latex.format_key_for_latex(step) + "} ")

            for sub_step in runtime_breakdown[step]:
                for stat in runtime_breakdown[step][sub_step]:
                    all_times[stat] += runtime_breakdown[step][sub_step][stat]

                total_wall_times = runtime_breakdown[step][sub_step]['total_wall_time']
                frame_times = runtime_breakdown[step][sub_step]['frame_time']

                latex_lines.append(f" & {Latex.format_key_for_latex(sub_step)} & "
                                   f"{Latex.to_mean_stddev(total_wall_times, formatter=Latex.format_timedelta)} & "
                                   f"{Latex.to_mean_stddev(frame_times, formatter=Latex.sec_to_ms)} \\\\")

        latex_lines.append(r"\midrule")
        latex_lines.append(
            f"\\textbf{{Total}} & & {Latex.to_mean_stddev(total_runtimes, formatter=Latex.format_timedelta)} & "
            f"{Latex.to_mean_stddev(total_frame_times, formatter=Latex.sec_to_ms)} \\\\")
        latex_lines.append(r"\bottomrule")
        latex_lines.append(r"\end{tabular}")

        runtime_breakdown_path = pjoin(self.latex_path, 'runtime_breakdown.tex')

        with open(runtime_breakdown_path, 'w') as f:
            f.write('\n'.join(latex_lines))

        logging.info(f"Exported runtime breakdown to {runtime_breakdown_path}.")

        # Per dataset performance stats
        latex_lines = [
            r"\begin{tabular}{llrrr}",
            r"\toprule",
            r"Dataset & Config & Wall Time (mm:ss) & RAM (GB) & VRAM (GB) \\",
        ]

        all_wall_times = []
        all_ram_usage = []
        all_vram_usage = []

        for dataset_name in self.dataset_names:
            latex_lines.append(r"\midrule")
            latex_lines.append(r"\multirow{3}{*}{" + Latex.format_dataset_name(dataset_name) + "}")

            for label in self.labels:
                total_time = performance_statistics[dataset_name][label]['total_time']
                ram_usage = performance_statistics[dataset_name][label]['ram_usage']
                vram_usage = performance_statistics[dataset_name][label]['vram_usage']

                latex_lines.append(
                    f" & {label} & {Latex.format_timedelta(total_time)} & {Latex.bytes_to_gigabytes(ram_usage)} & "
                    f"{Latex.bytes_to_gigabytes(vram_usage)} \\\\"
                )

                all_wall_times.append(total_time)
                all_ram_usage.append(ram_usage)
                all_vram_usage.append(vram_usage)

        latex_lines.append(r"\midrule")
        latex_lines.append(f"Average & & {Latex.to_mean_stddev(all_wall_times, formatter=Latex.format_timedelta)} & "
                           f"{Latex.to_mean_stddev(all_ram_usage, formatter=Latex.bytes_to_gigabytes)} & "
                           f"{Latex.to_mean_stddev(all_vram_usage, formatter=Latex.bytes_to_gigabytes)} \\\\")
        latex_lines.append(r"\bottomrule")
        latex_lines.append(r"\end{tabular}")

        performance_latex_path = pjoin(self.latex_path, 'per_dataset_performance.tex')

        with open(performance_latex_path, 'w') as f:
            f.write('\n'.join(latex_lines))

        logging.info(f"Exported per-dataset performance stats to {performance_latex_path}.")

        # Compression Statistics
        latex_lines = [
            r"\begin{tabular}{lrrrrrr}",
            r"\toprule",
            r"Layer & Meshes & Time (ms) & Size Before (MB) & Size After (MB) & Data Savings & Compression Ratio \\",
            r"\midrule"
        ]

        all_mesh_count = []
        all_times = []
        all_uncompressed_file_size = []
        all_compressed_file_size = []
        all_data_saving = []
        all_compression_ratio = []

        for layer in ('foreground', 'background'):
            stats = file_statistics_by_layer[layer]
            mesh_count = stats['mesh_count']
            time = stats['time']
            uncompressed_file_size = stats['uncompressed_file_size']
            compressed_file_size = stats['compressed_file_size']
            data_saving = stats['data_saving']
            compression_ratio = stats['compression_ratio']

            latex_lines.append(
                f"{layer.capitalize()} & "
                f"{Latex.to_mean_stddev(mesh_count, Latex.format_one_dp)} & "
                f"{Latex.to_mean_stddev(time, Latex.format_one_dp)} & "
                f"{Latex.to_mean_stddev(uncompressed_file_size, Latex.bytes_to_megabytes)} & "
                f"{Latex.to_mean_stddev(compressed_file_size, Latex.bytes_to_megabytes)} & "
                f"{Latex.to_mean_stddev(data_saving, Latex.percent_formatter)} & "
                f"{Latex.to_mean_stddev(compression_ratio, '{:,.2f}:1'.format)} \\\\"
            )

            all_mesh_count += mesh_count
            all_times += time
            all_uncompressed_file_size += uncompressed_file_size
            all_compressed_file_size += compressed_file_size
            all_data_saving += data_saving
            all_compression_ratio += compression_ratio

        latex_lines.append(r"\midrule")
        latex_lines.append(f"Average & {Latex.to_mean_stddev(all_mesh_count, formatter=Latex.format_one_dp)} & "
                           f"{Latex.to_mean_stddev(all_times, formatter=Latex.format_one_dp)} & "
                           f"{Latex.to_mean_stddev(all_uncompressed_file_size, formatter=Latex.bytes_to_megabytes)} & "
                           f"{Latex.to_mean_stddev(all_compressed_file_size, formatter=Latex.bytes_to_megabytes)} & "
                           f"{Latex.to_mean_stddev(all_data_saving, formatter=Latex.percent_formatter)} & "
                           f"{Latex.to_mean_stddev(all_compression_ratio, formatter='{:,.2f}:1'.format)} \\\\")
        latex_lines.append(r"\bottomrule")
        latex_lines.append(r"\end{tabular}")

        compression_latex_path = pjoin(self.latex_path, 'compression.tex')

        with open(compression_latex_path, 'w') as f:
            f.write('\n'.join(latex_lines))

        logging.info(f"Exported compression stats to {compression_latex_path}.")

    def run_trajectory_experiments(self):
        logging.info("Running trajectory comparisons...")
        trajectory_results = dict()

        for dataset_name in self.dataset_names:
            dataset_gt = HiveDataset(pjoin(self.output_path, f"{dataset_name}_{self.gt_label}"))
            frame_set = dataset_gt.select_key_frames(threshold=self.mesh_options.key_frame_threshold,
                                                     frame_step=self.frame_step)

            for label in (self.cm_label, self.est_label):
                dataset = HiveDataset(pjoin(self.output_path, f"{dataset_name}_{label}"))

                logging.info(f"Running trajectory comparison for dataset '{dataset_name}' and config '{label}'.")
                run_trajectory_comparisons(dataset, pred_trajectory=dataset.camera_trajectory,
                                           gt_trajectory=dataset_gt.camera_trajectory, dataset_name=dataset_name,
                                           pred_label=label, gt_label=self.gt_label,
                                           results_dict=trajectory_results,
                                           output_folder=self.trajectory_outputs_path,
                                           background_mesh_options=BackgroundMeshOptions(),
                                           frame_set=frame_set)

            with open(self.trajectory_results_path, 'w') as f:
                json.dump(trajectory_results, f)

    def export_trajectory_results(self):
        with open(self.trajectory_results_path, 'r') as f:
            trajectory_results = json.load(f)

        latex_lines = [
            r"\begin{tabular}{lllll}",
            r"\toprule",
            r"Dataset & Config & \acrshort{rpe}$_r$ (\degree) & \acrshort{rpe}$_t$ (cm) & \acrshort{ate} (cm) \\"
        ]

        all_data = {
            label: {
                'rpe': {
                    'rotation': [],
                    'translation': []
                },
                'ate': []
            }
            for label in (self.cm_label, self.est_label)
        }

        def format_percent(number):
            if number < -0.0001:
                colour = 'Green'
            elif number > 0.0001:
                colour = 'BrickRed'
            else:
                colour = 'black'
                # Ensure there's no sign
                number = abs(number)

            return f"(\\textcolor{{{colour}}}{{{number * 100:,.2f}\%}})"

        for dataset_name in self.dataset_names:
            latex_lines.append(r"\midrule")
            latex_lines.append(f"\\multirow{{2}}{{*}}{{{Latex.format_dataset_name(dataset_name)}}}")

            for label in (self.cm_label, self.est_label):
                row = trajectory_results[dataset_name][label]

                rpe_rotation = row['rpe']['rotation']
                rpe_translation = row['rpe']['translation'] * 100  # convert from meters to centimeters
                ate = row['ate'] * 100  # convert from meters to centimeters

                if label == self.est_label:
                    cm_stats = trajectory_results[dataset_name][self.cm_label]
                    rpe_rotation_percent_change = rpe_rotation / cm_stats['rpe']['rotation'] - 1
                    rpe_translation_percent_change = rpe_translation / (100 * cm_stats['rpe']['translation']) - 1
                    ate_percent_change = ate / (100 * cm_stats['ate']) - 1

                    latex_lines.append(f" & {label}"
                                       f" & {rpe_rotation:,.2f} {format_percent(rpe_rotation_percent_change)}"
                                       f" & {rpe_translation:,.2f} {format_percent(rpe_translation_percent_change)}"
                                       f" & {ate:,.2f} {format_percent(ate_percent_change)} \\\\")
                else:
                    latex_lines.append(
                        f" & {label} & {rpe_rotation:<9,.2f} & {rpe_translation:<9,.2f} & {ate:<9,.2f} \\\\")

                all_data[label]['rpe']['rotation'].append(rpe_rotation)
                all_data[label]['rpe']['translation'].append(rpe_translation)
                all_data[label]['ate'].append(ate)

        latex_lines.append(r"\midrule")
        latex_lines.append(r"\multirow{3}{*}{\textbf{Mean}}")

        for label in (self.cm_label, self.est_label):
            if label == self.est_label:
                est_stats = all_data[self.est_label]
                cm_stats = all_data[self.cm_label]

                rpe_rotation = statistics.mean(est_stats['rpe']['rotation'])
                rpe_translation = statistics.mean(est_stats['rpe']['translation'])
                ate = statistics.mean(est_stats['ate'])

                rpe_rotation_cm = statistics.mean(cm_stats['rpe']['rotation'])
                rpe_translation_cm = statistics.mean(cm_stats['rpe']['translation'])
                ate_cm = statistics.mean(cm_stats['ate'])

                rpe_rotation_percent_change = rpe_rotation / rpe_rotation_cm - 1
                rpe_translation_percent_change = rpe_translation / rpe_translation_cm - 1
                ate_percent_change = ate / ate_cm - 1

                latex_lines.append(f" & {label}"
                                   f" & {rpe_rotation:,.2f} {format_percent(rpe_rotation_percent_change)}"
                                   f" & {rpe_translation:,.2f} {format_percent(rpe_translation_percent_change)}"
                                   f" & {ate:,.2f} {format_percent(ate_percent_change)} \\\\")
            else:
                latex_lines.append(f" & {label} & {Latex.to_mean(all_data[label]['rpe']['rotation']):<9} & "
                                   f"{Latex.to_mean(all_data[label]['rpe']['translation']):<9} & "
                                   f"{Latex.to_mean(all_data[label]['ate']):<9} \\\\")

        latex_lines.append(
            f" & all & {Latex.to_mean(all_data[self.cm_label]['rpe']['rotation'] + all_data[self.est_label]['rpe']['rotation']):<9} & "
            f"{Latex.to_mean(all_data[self.cm_label]['rpe']['translation'] + all_data[self.est_label]['rpe']['translation']):<9} & "
            f"{Latex.to_mean(all_data[self.cm_label]['ate'] + all_data[self.est_label]['ate']):<9} \\\\")
        latex_lines.append(r"\bottomrule")
        latex_lines.append(r"\end{tabular}")

        trajectory_latex_path = pjoin(self.latex_path, 'trajectory.tex')

        with open(trajectory_latex_path, 'w') as f:
            f.write('\n'.join(latex_lines))

        logging.info(f"Exported trajectory stats to {trajectory_latex_path}.")

    def run_bundlefusion_experiments(self):
        logging.info("Running reconstruction comparisons...")
        recon_folder = pjoin(self.output_path, 'reconstruction')

        bundle_fusion_results = dict()

        for dataset_name in self.dataset_names:
            bundle_fusion_results[dataset_name] = dict()

            for label in (self.gt_label, self.est_label):
                dataset = HiveDataset(pjoin(self.output_path, f"{dataset_name}_{label}"))
                frame_set = dataset.select_key_frames(threshold=self.mesh_options.key_frame_threshold,
                                                      frame_step=self.frame_step)

                logging.info(f"Running comparisons for dataset '{dataset_name}' and config '{label}'...")
                mesh_output_path = pjoin(recon_folder, dataset_name, label)
                os.makedirs(mesh_output_path, exist_ok=True)

                logging.info('Creating TSDFFusion mesh...')
                tsdf_mesh = tsdf_fusion(dataset, self.mesh_options, frame_set=frame_set)
                tsdf_mesh.export(pjoin(mesh_output_path, "tsdf.ply"))

                # This is needed in case BundleFusion has already been run with the dataset.
                logging.info('Creating BundleFusion mesh...')
                dataset.overwrite_ok = self.overwrite_ok

                try:
                    bf_mesh = bundle_fusion(output_folder='bundle_fusion', dataset=dataset, options=self.mesh_options)
                    bf_mesh.export(pjoin(mesh_output_path, 'bf.ply'))
                    bundle_fusion_results[dataset_name][label] = True
                except RuntimeError as e:
                    logging.warning(f"Encountered error while running BundleFusion: {e}")
                    bundle_fusion_results[dataset_name][label] = False

                if label != self.gt_label and (mesh := tsdf_fusion_with_colmap(dataset, frame_set, self.mesh_options)):
                    logging.info('Creating TSDFFusion mesh with COLMAP depth...')
                    mesh.export(pjoin(mesh_output_path, 'colmap_depth.ply'))

        with open(self.bundle_fusion_results_path, 'w') as f:
            json.dump(bundle_fusion_results, f)

        logging.info(f"Saved BundleFusion results to {self.bundle_fusion_results_path}.")

    def export_bundle_fusion_results(self):
        with open(self.bundle_fusion_results_path, 'r') as f:
            bundle_fusion_results = json.load(f)

        latex_lines = [
            r"\begin{tabular}{llll}",
            r"\toprule",
            r"Dataset & Config & \multicolumn{2}{c}{Produced Mesh?} \\",
            r"        &        & BundleFusion & HIVE \\",
            r"\midrule",
        ]

        successes = {
            self.gt_label: 0,
            self.est_label: 0
        }

        for dataset_name in bundle_fusion_results:
            latex_lines.append(rf"\multirow{{2}}{{*}}{{{Latex.format_dataset_name(dataset_name)}}}")

            for label in bundle_fusion_results[dataset_name]:
                produced_mesh = bundle_fusion_results[dataset_name][label]

                if produced_mesh:
                    successes[label] += 1

                symbol_text = r"\cmark" if produced_mesh else r"\xmark"
                row_text = rf" & {label.upper()} & {symbol_text} & \cmark \\"
                latex_lines.append(row_text)

            latex_lines.append(r"\midrule")

        item_count = len(bundle_fusion_results)
        bf_success_rates = {label: successes[label] / item_count for label in (self.gt_label, self.est_label)}

        latex_lines.append(rf"All & {self.gt_label.upper()} & {bf_success_rates[self.gt_label] * 100:.0f}\% & 100\% \\")
        latex_lines.append(
            rf"All & {self.est_label.upper()} & {bf_success_rates[self.est_label] * 100:.0f}\% & 100\% \\")
        latex_lines.append(r"\bottomrule")
        latex_lines.append(r"\end{tabular}")

        bundle_fusion_latex_path = pjoin(self.latex_path, 'bundle_fusion.tex')

        with open(bundle_fusion_latex_path, 'w') as f:
            f.write('\n'.join(latex_lines))

        logging.info(f"Exported Bundle Fusion experiment results to {bundle_fusion_latex_path}.")

    def export_latex_preamble(self):
        """Create a latex file that contains the package imports and commands necessary for rendering the tables from
        this script."""
        latex_lines = [
            r"\usepackage{booktabs}  % For \toprule, \midrule and \bottomrule commands",
            r"\usepackage{multirow}  % Multi-row table cells",
            r"\usepackage{pifont}  % Various symbols",
            r"\newcommand{\cmark}{\ding{51}}  % A checkmark/tick",
            r"\newcommand{\xmark}{\ding{55}}  % A cross mark"
        ]

        preamble_path = pjoin(self.latex_path, 'preamble.tex')

        with open(preamble_path, 'w') as f:
            f.write('\n'.join(latex_lines))

        logging.info(f"Wrote Latex preamble to {preamble_path}.")

    def run_compression_experiments(self):
        config = self.mesh_compression_experiment_config

        for dataset_name in self.dataset_names:
            for label, pipeline_config in self.pipeline_configurations:
                dataset_name_and_label = f"{dataset_name}_{label}"

                logging.info(f"Running Compression Comparison for {dataset_name_and_label}...")

                dataset = HiveDataset(pjoin(self.output_path, dataset_name_and_label))
                output_path = pjoin(self.compression_outputs_path, dataset_name_and_label)
                uncompressed_mesh_output_folder = os.path.join(output_path, config.uncompressed_mesh_folder)
                compressed_mesh_output_folder = os.path.join(output_path, config.compressed_mesh_folder)

                if (os.path.isdir(uncompressed_mesh_output_folder) and os.path.isdir(compressed_mesh_output_folder) and
                        not self.overwrite_ok):
                    logging.info(f"Found cached data for {dataset_name_and_label}, skipping.")
                    continue

                storage_options = StorageOptions(dataset_path=dataset.base_path,
                                                 output_path=output_path,
                                                 overwrite_ok=False)
                pipeline = Pipeline(options=pipeline_config, storage_options=storage_options,
                                    static_mesh_options=self.mesh_options)

                fg_mesh = pipeline.process_frame(dataset, index=0)
                bg_mesh = pipeline.create_static_mesh(dataset, options=self.mesh_options)
                os.makedirs(uncompressed_mesh_output_folder, exist_ok=True)
                os.makedirs(compressed_mesh_output_folder, exist_ok=True)

                for layer, mesh in ((config.fg_mesh_name, fg_mesh), (config.bg_mesh_name, bg_mesh)):
                    if mesh.is_empty:
                        logging.info(f"Mesh for {layer} is empty, skipping.")
                        continue

                    # noinspection PyUnresolvedReferences
                    if isinstance(mesh.visual, trimesh.visual.TextureVisuals):
                        # Use vertex colours instead of textures since they seem to get lost for some reason?
                        mesh.visual = mesh.visual.to_color()

                    uncompressed_mesh_path = os.path.join(output_path, config.uncompressed_mesh_folder,
                                                          f"{layer}{config.uncompressed_mesh_extension}")
                    compressed_mesh_path = os.path.join(output_path, config.compressed_mesh_folder,
                                                        f"{layer}{config.compressed_mesh_extension}")
                    logging.info(f"Exporting {layer} mesh to {uncompressed_mesh_path} and {compressed_mesh_path}...")

                    mesh.export(uncompressed_mesh_path)

                    with open(compressed_mesh_path, 'wb') as f:
                        # noinspection PyUnresolvedReferences
                        f.write(trimesh.exchange.ply.export_draco(mesh))

        self._render_compressed_mesh_comparison()

    def _render_compressed_mesh_comparison(self):
        config = self.mesh_compression_experiment_config

        def load_draco(path: str) -> trimesh.Trimesh:
            with open(path, 'rb') as f:
                # noinspection PyUnresolvedReferences
                mesh_data = trimesh.exchange.ply.load_draco(f)

            return trimesh.Trimesh(**mesh_data)

        lpips_fn = LPIPS(net='alex')
        results: Dict[str, Dict[str, Dict[str, float]]] = dict()
        compression_configs = ((config.uncompressed_mesh_folder, config.uncompressed_mesh_extension, trimesh.load),
                               (config.compressed_mesh_folder, config.compressed_mesh_extension, load_draco))

        for dataset_name in self.dataset_names:
            results[dataset_name] = {}

            for label in self.labels:
                dataset_name_and_label = f"{dataset_name}_{label}"
                logging.info(f"Running Compression Comparison for {dataset_name_and_label}...")

                image_pair = []

                for folder, ext, load_fn in compression_configs:
                    screen_capture_path = os.path.join(self.compression_outputs_path, dataset_name_and_label,
                                                       f"{folder}.png")

                    if os.path.isfile(screen_capture_path) and not self.overwrite_ok:
                        logging.info(f"Found cached result at {screen_capture_path}, skipping {folder}...")
                        image_pair.append(cv2.imread(screen_capture_path))
                        continue

                    mesh_folder = os.path.join(self.compression_outputs_path, dataset_name_and_label, folder)
                    fg_mesh_path = os.path.join(mesh_folder, f"{config.fg_mesh_name}{ext}")
                    bg_mesh_path = os.path.join(mesh_folder, f"{config.bg_mesh_name}{ext}")

                    if os.path.isfile(fg_mesh_path):
                        fg_mesh = load_fn(fg_mesh_path)
                    else:
                        fg_mesh = trimesh.Trimesh()

                    bg_mesh = load_fn(bg_mesh_path)

                    scene = trimesh.Scene()
                    scene.add_geometry(fg_mesh)
                    scene.add_geometry(bg_mesh)
                    # This rotation corrects for the rotation applied for viewing in the web-based renderer.
                    rotation_matrix = trimesh.transformations.rotation_matrix(angle=math.pi, direction=[1, 0, 0])
                    scene.apply_transform(rotation_matrix)
                    scene.camera_transform = scene.camera.look_at(bg_mesh.vertices)

                    screen_capture = scene.save_image(resolution=(640, 480))

                    with open(screen_capture_path, 'wb') as f:
                        f.write(screen_capture)

                    image_pair.append(cv2.imread(screen_capture_path))

                logging.info(f"Calculating image similarity for {dataset_name_and_label}...")
                ssim, psnr, lpips = compare_images(image_pair[0], image_pair[1], lpips_fn=lpips_fn)
                results[dataset_name][label] = {
                    'ssim': ssim,
                    'psnr': psnr,
                    'lpips': lpips
                }
                logging.info(results[dataset_name][label])

        with open(self.compression_results_path, 'w') as f:
            json.dump(results, f)

    def export_compression_results(self):
        with open(self.compression_results_path, 'r') as f:
            results = json.load(f)

        latex_lines = [
            r"\begin{tabular}{llrrr}",
            r"\toprule",
            r"Dataset & Config & SSIM & PSNR & LPIPS \\",
            r"\midrule"
        ]

        metrics = 'ssim', 'psnr', 'lpips'

        averages = {
            label: {
                metric_name: {
                    'sum': 0.0,
                    'count': 0,
                }
                for metric_name in metrics
            }
            for label in self.labels
        }

        for dataset_name in results:
            latex_lines.append(rf"\multirow{{3}}{{*}}{{{Latex.format_dataset_name(dataset_name)}}}")

            for label in results[dataset_name]:
                ssim, psnr, lpips = results[dataset_name][label].values()
                latex_lines.append(rf"& {label} & {ssim:.2f} & {psnr:.2f} & {lpips:.2f} \\")

                for metric_name in metrics:
                    averages[label][metric_name]['sum'] += results[dataset_name][label][metric_name]
                    averages[label][metric_name]['count'] += 1

            latex_lines.append(r"\midrule")

        average_all = {metric_name: 0.0 for metric_name in metrics}
        latex_lines.append(rf"\multirow{{4}}{{*}}{{Mean}} ")

        for label in averages:
            parts = [label]

            for metric_name in metrics:
                mean = averages[label][metric_name]['sum'] / averages[label][metric_name]['count']
                parts.append(f"{mean:.2f}")
                average_all[metric_name] += mean / len(averages)

            latex_lines.append(rf" & {' & '.join(parts)} \\")

        latex_lines.append(
            rf" & All & {average_all['ssim']:.2f} & {average_all['psnr']:.2f} & {average_all['lpips']:.2f} \\")
        latex_lines.append(r"\bottomrule")
        latex_lines.append(r"\end{tabular}")

        latex_code = '\n'.join(latex_lines)
        latex_path = os.path.join(self.latex_path, "compression_image_comparison.tex")

        with open(latex_path, "w") as f:
            f.write(latex_code)

        logging.info(f"Exported compression image similarity experiment results to {latex_path}.")

    def run_inpainting_experiments(self):
        logging.info("Running Inpainting Comparisons...")

        logging.debug(f"Loading inpainting model...")
        model = InpaintingExperiment.load_inpainting_model()
        lpips_fn = LPIPS(net='alex')
        results = dict()

        for dataset_name in self.dataset_names:
            results[dataset_name] = dict()

            for label in (self.gt_label, self.est_label):
                dataset_name_and_label = f"{dataset_name}_{label}"
                logging.info(f"Running Inpainting Comparison for {dataset_name_and_label}...")

                dataset_path = pjoin(self.output_path, dataset_name_and_label)
                logging.debug(f"Loading dataset from {dataset_path}...")
                dataset = HiveDataset(dataset_path)
                rgb_raw = dataset.rgb_dataset[0]
                depth_raw = dataset.depth_dataset[0]
                mask = dataset.mask_dataset[0] > 0

                output_path = pjoin(self.inpainting_outputs_path, dataset_name_and_label)
                logging.debug(f"Copying reference frame data to {output_path}...")
                os.makedirs(output_path, exist_ok=True)

                Image.fromarray(rgb_raw).save(os.path.join(output_path, 'rgb.png'))
                Image.fromarray(InpaintingExperiment.depth_to_img(depth_raw)).save(
                    os.path.join(output_path, 'depth.png'))
                Image.fromarray(InpaintingExperiment.mask_to_img(mask), mode='L').save(
                    os.path.join(output_path, 'mask.png'))

                logging.debug(f"Looping over crop regions...")
                crop_regions = InpaintingExperiment.get_crop_regions(rgb_frame=rgb_raw, binary_mask=mask)
                results[dataset_name][label] = []

                # TODO: Also include before & after comparison of rendered meshes?

                for crop_index, crop_mask in enumerate(crop_regions):
                    logging.debug(f"Processing crop region {crop_index}...")
                    rgb_inpainted = InpaintingExperiment.inpaint_rgb(model=model, image=rgb_raw, mask=crop_mask)
                    depth_inpainted = InpaintingExperiment.inpaint_depth(depth_raw, crop_mask,
                                                                         depth_scale=dataset.depth_scaling_factor)

                    filename_suffix = HiveDataset.index_to_filename(crop_index, file_extension='png')
                    Image.fromarray(InpaintingExperiment.mask_to_img(crop_mask)).save(
                        os.path.join(output_path, f"mask_crop_{filename_suffix}"))
                    Image.fromarray(rgb_inpainted).save(os.path.join(output_path, f"rgb_inpainted_{filename_suffix}"))
                    Image.fromarray(InpaintingExperiment.depth_to_img(depth_inpainted)).save(
                        os.path.join(output_path, f"depth_inpainted_{filename_suffix}"))

                    ssim, psnr, lpips = InpaintingExperiment.compare_rgb(rgb_raw, rgb_inpainted, crop_mask,
                                                                         lpips_fn=lpips_fn)
                    depth_metrics = InpaintingExperiment.compare_depth(depth_raw, depth_inpainted, crop_mask)

                    crop_results = {
                        'image_similarity': {
                            'ssim': ssim,
                            'psnr': psnr,
                            'lpips': lpips
                        },
                        'depth_metrics': depth_metrics
                    }
                    results[dataset_name][label].append(crop_results)

                    crop_results_path = os.path.join(
                        output_path, f"results_{HiveDataset.index_to_filename(crop_index, file_extension='json')}"
                    )

                    with open(crop_results_path, 'w') as f:
                        json.dump(crop_results, f)

        logging.info(f"Saving inpainting comparison results to {self.inpainting_results_path}...")

        with open(self.inpainting_results_path, 'w') as f:
            json.dump(results, f)

    def export_inpainting_results(self):
        with open(self.inpainting_results_path, 'r') as f:
            inpainting_results = json.load(f)

        ssim_key = 'ssim'
        psnr_key = 'psnr'
        lpips_key = 'lpips'
        rmse_key = 'rmse'
        abs_rel_key = 'abs_rel'
        delta_1_key = 'delta_1'

        class Average:
            def __init__(self):
                self.total = 0.0
                self.count = 0

            def update(self, value):
                if np.isnan(value):
                    return

                self.total += value
                self.count += 1

            @property
            def mean(self):
                return self.total / self.count

        averages = {
            dataset_name: {
                label: {
                    ssim_key: Average(),
                    psnr_key: Average(),
                    lpips_key: Average(),
                    rmse_key: Average(),
                    abs_rel_key: Average(),
                    delta_1_key: Average()
                }
                for label in (self.gt_label, self.est_label)
            }
            for dataset_name in self.dataset_names
        }

        average_by_label = {
            label: {
                ssim_key: Average(),
                psnr_key: Average(),
                lpips_key: Average(),
                rmse_key: Average(),
                abs_rel_key: Average(),
                delta_1_key: Average()
            }
            for label in (self.gt_label, self.est_label)
        }

        for dataset_name in inpainting_results:
            for label in (self.gt_label, self.est_label):
                for row in inpainting_results[dataset_name][label]:
                    image_similarity = row['image_similarity']
                    depth_metrics = row['depth_metrics']

                    averages[dataset_name][label][ssim_key].update(image_similarity[ssim_key])
                    averages[dataset_name][label][psnr_key].update(image_similarity[psnr_key])
                    averages[dataset_name][label][lpips_key].update(image_similarity[lpips_key])
                    averages[dataset_name][label][rmse_key].update(depth_metrics[rmse_key])
                    averages[dataset_name][label][abs_rel_key].update(depth_metrics[abs_rel_key])
                    averages[dataset_name][label][delta_1_key].update(depth_metrics[delta_1_key])

                    average_by_label[label][ssim_key].update(image_similarity[ssim_key])
                    average_by_label[label][psnr_key].update(image_similarity[psnr_key])
                    average_by_label[label][lpips_key].update(image_similarity[lpips_key])
                    average_by_label[label][rmse_key].update(depth_metrics[rmse_key])
                    average_by_label[label][abs_rel_key].update(depth_metrics[abs_rel_key])
                    average_by_label[label][delta_1_key].update(depth_metrics[delta_1_key])

        latex_lines = [
            r"\begin{tabular}{llrrrrrr}",
            r"\toprule",
            r" & & \multicolumn{3}{c}{Image Similarity} & \multicolumn{3}{c}{Depth Metrics} \\",
            r"\cmidrule(lr){3-5} \cmidrule(lr){6-8}",
            r"Dataset & Config & SSIM $\uparrow$ & PSNR $\uparrow$ & LPIPS $\downarrow$ & "
            r"RMSE $\downarrow$ & absRel $\downarrow$ & $\delta^1$ $\uparrow$ \\",
            r"\midrule"
        ]

        for dataset_name in averages:
            latex_lines.append(rf"\multirow{{2}}{{*}}{{{Latex.format_dataset_name(dataset_name)}}}")

            for label in averages[dataset_name]:
                data = averages[dataset_name][label]
                latex_lines.append(rf" & {label.upper()} & {data[ssim_key].mean:.2f} & {data[psnr_key].mean:.2f} & "
                                   rf"{data[lpips_key].mean:.2f} & {data[rmse_key].mean:.2f} & "
                                   rf"{data[abs_rel_key].mean:.2f} & {data[delta_1_key].mean:.2f} \\")

            latex_lines.append(r"\midrule")

        latex_lines.append(rf"\multirow{{2}}{{*}}{{Mean}}")

        for label in average_by_label:
            data = average_by_label[label]
            latex_lines.append(rf" & {label.upper()} & {data[ssim_key].mean:.2f} & {data[psnr_key].mean:.2f} & "
                               rf"{data[lpips_key].mean:.2f} & {data[rmse_key].mean:.2f} & "
                               rf"{data[abs_rel_key].mean:.2f} & {data[delta_1_key].mean:.2f} \\")

        latex_lines.append(r"\bottomrule")
        latex_lines.append(r"\end{tabular}")

        latex_path = os.path.join(self.latex_path, 'inpainting.tex')

        with open(latex_path, 'w') as f:
            f.write('\n'.join(latex_lines))

        logging.info(f"Exported inpainting latex table to {latex_path}.")

    def run_llff_experiments(self, sequence_names: List[str]):
        llff_folder = os.path.join(self.output_path, 'llff')
        lpips_fn = LPIPS(net='alex')

        for sequence_name in sequence_names:
            results_path = os.path.join(llff_folder, sequence_name)

            if not self.overwrite_ok and os.path.isdir(results_path) and len(os.listdir(results_path)) > 0:
                logging.info(f"Cached results found for {sequence_name} in {results_path}, skipping...")
                continue

            os.makedirs(results_path, exist_ok=True)

            LLFFExperiment.compare_renders(data_folder=self.data_path,
                                           sequence_name=sequence_name,
                                           output_folder=os.path.join(self.output_path, 'converted_dataset'),
                                           results_folder=results_path,
                                           frame_index=0,
                                           lpips_fn=lpips_fn,
                                           no_cache=self.overwrite_ok)

        results_records = LLFFExperiment.gather_results(output_folder=llff_folder)
        LLFFExperiment.export_latex(results_records, summary_path=self.summaries_path, latex_path=self.latex_path)

    def run_hypernerf_experiments(self):
        for sequence_name in HyperNeRFExperiments.sequence_names:
            HyperNeRFExperiments.fetch_sequence(self.data_path, sequence_name)

            adaptor = HyperNeRFAdaptor(
                base_path=os.path.join(self.data_path, sequence_name),
                output_path=os.path.join(self.output_path, 'converted_dataset', sequence_name),
                num_frames=self.num_frames,
                frame_step=self.frame_step,
                colmap_options=self.colmap_options,
            )

            dataset = adaptor.convert(estimate_pose=False, estimate_depth=True, inpainting_mode=self.inpainting_mode,
                                      static_camera=False, no_cache=self.overwrite_ok)
            pipeline = Pipeline(self.cm_options, storage_options=StorageOptions(dataset.base_path,
                                                                                output_path=dataset.base_path,
                                                                                overwrite_ok=self.overwrite_ok,
                                                                                no_cache=self.overwrite_ok))

            fg = pipeline.process_frame(dataset, index=0)
            bg = pipeline.create_static_mesh(dataset)

            camera_matrix = adaptor.get_camera_matrix_wrapper(index=0)
            pose = adaptor.get_pose(index=0, is_train=False)

            rgb = LLFFExperiment.render_mesh(camera_matrix, pose, fg, bg)
            val_frame = adaptor.get_frame(index=0, is_train=False)

            output_side_by_side = np.hstack((rgb, val_frame))
            Image.fromarray(output_side_by_side).save('image.jpg')

            break


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_path', help='The path to save the experiment outputs and results to.',
                        required=True, type=str)
    parser.add_argument('--data_path', help='The path to the folder containing the datasets.',
                        required=True, type=str)
    parser.add_argument('--log_file', type=str,
                        help='The path to save the logs to.', default='experiments.log')
    parser.add_argument('-y', dest='overwrite_ok', action='store_true',
                        help='Whether to overwrite any old results.')
    args = parser.parse_args()

    log_file = args.log_file
    setup_logger(log_file)

    logging.info(f"Running experiments with arguments: {args}")

    # TODO: Make these configurable via CLI.
    dataset_names = ['rgbd_dataset_freiburg3_walking_xyz',
                     'rgbd_dataset_freiburg3_sitting_xyz',
                     'garden',
                     'small_tree']

    experiments = Experiments(output_path=args.output_path, data_path=args.data_path,
                              overwrite_ok=args.overwrite_ok,
                              dataset_names=dataset_names,
                              num_frames=800,
                              frame_step=15,
                              log_file=log_file)

    with virtual_display():
        experiments.run_kid_running_experiments(filename='kid_running.mp4')
        experiments.run_pipeline_experiments()
        experiments.export_pipeline_results()
        experiments.run_trajectory_experiments()
        experiments.export_trajectory_results()
        experiments.run_bundlefusion_experiments()
        experiments.export_bundle_fusion_results()
        experiments.export_latex_preamble()
        experiments.run_compression_experiments()
        experiments.export_compression_results()
        experiments.run_inpainting_experiments()
        experiments.export_inpainting_results()

        experiments.run_llff_experiments(LLFFExperiment.sequence_names)

        experiments.run_hypernerf_experiments()


if __name__ == '__main__':
    main()
