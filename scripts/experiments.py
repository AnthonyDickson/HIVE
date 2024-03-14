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
from contextlib import contextmanager
from dataclasses import dataclass
from os.path import join as pjoin
from typing import Tuple, Dict, Optional, List, Union, Callable

import cv2
import numpy as np
import torch
import trimesh
import yaml
from PIL import Image
from lpips import LPIPS
from omegaconf import OmegaConf
from torch.utils.data.dataloader import default_collate

from hive.dataset import CMUPanopticDataset
from hive.fusion import tsdf_fusion, bundle_fusion
from hive.geometric import Trajectory, point_cloud_from_rgbd
from hive.io import HiveDataset, temporary_trajectory, DatasetMetadata
from hive.options import BackgroundMeshOptions, PipelineOptions, StorageOptions, InpaintingMode, COLMAPOptions, \
    WebXROptions
from hive.pipeline import Pipeline
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


class PanopticDatasetExperiment:
    @classmethod
    def create_ply(cls, dataset: CMUPanopticDataset, kinect_node: int, frame_index: int = 0):
        calibration = dataset.kinect_calibration[kinect_node]
        rgb, depth = dataset.get_synced_frame_data(frame_index=frame_index, kinect_node=kinect_node)
        depth = cv2.resize(depth, dsize=tuple(reversed(rgb.shape[:2])), interpolation=cv2.INTER_NEAREST)

        vertices, colours = point_cloud_from_rgbd(rgb, depth, mask=np.ones(rgb.shape[:2], dtype=bool),
                                                  K=calibration.K_color)
        ply = trimesh.PointCloud(vertices=vertices, colors=colours)

        return ply


# TODO: Pull out each experiment type into own class.
# TODO: Add experiments for view reconstruction from Novel View Synthesis datasets. Record cam00 (reference view)
#  metrics separately from the rest.
class Experiments:
    def __init__(self, data_path: str, output_path: str, overwrite_ok: bool, dataset_names: List[str],
                 num_frames: int, frame_step: int, log_file: str):
        self.data_path = data_path
        self.output_path = output_path
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

        self.gt_options = PipelineOptions(num_frames=num_frames, frame_step=frame_step,
                                          estimate_pose=False, estimate_depth=False,
                                          inpainting_mode=InpaintingMode.Lama_Image_CV2_Depth, log_file=self.log_file)
        self.cm_options = PipelineOptions(num_frames=num_frames, frame_step=frame_step,
                                          estimate_pose=True, estimate_depth=False,
                                          inpainting_mode=InpaintingMode.Lama_Image_CV2_Depth, log_file=self.log_file)
        self.dpt_options = PipelineOptions(num_frames=num_frames, frame_step=frame_step,
                                           estimate_pose=False, estimate_depth=True,
                                           inpainting_mode=InpaintingMode.Lama_Image_CV2_Depth, log_file=self.log_file)
        self.est_options = PipelineOptions(num_frames=num_frames, frame_step=frame_step,
                                           estimate_pose=True, estimate_depth=True,
                                           inpainting_mode=InpaintingMode.Lama_Image_CV2_Depth, log_file=self.log_file)

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
            for dataset_name in dataset_names
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
                bg_mesh = pipeline.create_static_mesh(
                    dataset,
                    options=self.mesh_options,
                    frame_set=dataset.select_key_frames(threshold=self.mesh_options.key_frame_threshold,
                                                        frame_step=self.frame_step)
                )
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

        with virtual_display():
            for dataset_name in self.dataset_names:
                results[dataset_name] = {}

                for label in self.labels:
                    dataset_name_and_label = f"{dataset_name}_{label}"
                    logging.info(f"Running Compression Comparison for {dataset_name_and_label}...")

                    image_pair = []

                    for folder, ext, load_fn in compression_configs:
                        screen_capture_path = os.path.join(self.compression_outputs_path, dataset_name_and_label,
                                                           f"{folder}.png")

                        # if os.path.isfile(screen_capture_path) and not self.overwrite_ok:
                        #     logging.info(f"Found cached result at {screen_capture_path}, skipping {folder}...")
                        #     image_pair.append(cv2.imread(screen_capture_path))
                        #     continue

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
            for dataset_name in dataset_names
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


if __name__ == '__main__':
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
                     'small_tree',
                     'edwardsBay']

    experiments = Experiments(output_path=args.output_path, data_path=args.data_path,
                              overwrite_ok=args.overwrite_ok,
                              dataset_names=dataset_names,
                              num_frames=800,
                              frame_step=15,
                              log_file=log_file)

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

    dataset = CMUPanopticDataset(base_path=os.path.join(args.data_path, '170221_haggling_m3'))
    ply = PanopticDatasetExperiment.create_ply(dataset, kinect_node=1, frame_index=105)
    ply_path = os.path.join(args.output_path, 'haggling.ply')
    ply.export(ply_path)
