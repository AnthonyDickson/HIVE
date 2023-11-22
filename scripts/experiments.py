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
import datetime
import json
import logging
import os.path
import shutil
import statistics
from collections import defaultdict
from os.path import join as pjoin
from typing import Tuple, Dict, Optional, List, Union, Callable

import numpy as np
import pandas as pd
import trimesh

from video2mesh.fusion import tsdf_fusion, bundle_fusion
from video2mesh.geometric import Trajectory
from video2mesh.io import VTMDataset, temporary_trajectory, DatasetMetadata
from video2mesh.options import BackgroundMeshOptions, PipelineOptions, StorageOptions, InpaintingMode, COLMAPOptions, \
    WebXROptions
from video2mesh.pipeline import Pipeline
from video2mesh.utils import setup_logger, tqdm_imap, set_key_path


def setup(output_path: str, overwrite_ok: bool):
    if os.path.isdir(output_path) and not overwrite_ok:
        user_input = input(f"The output folder at {output_path} already exists, "
                           f"do you want to delete this folder before continuing? (y/n):")
        should_delete = user_input.lower() == 'y'

        if should_delete:
            shutil.rmtree(output_path)
        else:  # elif not overwrite_ok:
            raise RuntimeError(f"The output folder at {output_path} already exists. "
                               "Either change the output path or delete the existing folder.")

    os.makedirs(output_path, exist_ok=overwrite_ok)


def add_key(key, dataset_name, pred_label, results_dict):
    if key not in results_dict:
        results_dict[key] = dict()

    if dataset_name not in results_dict[key]:
        results_dict[key][dataset_name] = dict()

    if pred_label not in results_dict[key][dataset_name]:
        results_dict[key][dataset_name][pred_label] = dict()


def run_trajectory_comparisons(dataset, pred_trajectory: Trajectory, gt_trajectory: Trajectory, dataset_name: str,
                               pred_label: str, gt_label: str, results_dict: dict, output_folder: str,
                               background_mesh_options: BackgroundMeshOptions, frame_step=1):
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
        frame_set = list(range(0, dataset.num_frames, frame_step))

        if frame_set[-1] != dataset.num_frames - 1:
            frame_set.append(dataset.num_frames - 1)

        mesh = tsdf_fusion(dataset, background_mesh_options, frame_set=frame_set)
        mesh.export(pjoin(experiment_path, f"mesh.ply"))


def tsdf_fusion_with_colmap(dataset: VTMDataset, frame_set: List[int], mesh_options: BackgroundMeshOptions) -> Optional[
    trimesh.Trimesh]:
    depth_folder = pjoin(dataset.base_path, 'colmap_depth')
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

    metadata.save(pjoin(tmp_dir, VTMDataset.metadata_filename))
    np.savetxt(pjoin(tmp_dir, VTMDataset.camera_matrix_filename), camera_matrix)
    np.savetxt(pjoin(tmp_dir, VTMDataset.camera_trajectory_filename), poses)

    rgb_path = pjoin(tmp_dir, VTMDataset.rgb_folder)
    os.makedirs(rgb_path)

    def copy_rgb(index_filename):
        index, filename = index_filename
        src = pjoin(dataset.path_to_rgb_frames, filename)
        dst = pjoin(rgb_path, VTMDataset.index_to_filename(index))
        shutil.copy(src, dst)

    tqdm_imap(copy_rgb, list(enumerate(rgb_files)))

    def copy_depth(index_filename):
        index, filename = index_filename
        src = pjoin(depth_folder, filename)
        dst = pjoin(depth_path, VTMDataset.index_to_filename(index))
        shutil.copy(src, dst)

    depth_path = pjoin(tmp_dir, VTMDataset.depth_folder)
    os.makedirs(depth_path)
    tqdm_imap(copy_depth, list(enumerate(depth_files)))

    def copy_mask(index_filename):
        index, filename = index_filename
        src = pjoin(dataset.path_to_masks, filename)
        dst = pjoin(mask_path, VTMDataset.index_to_filename(index))
        shutil.copy(src, dst)

    mask_path = pjoin(tmp_dir, VTMDataset.mask_folder)
    os.makedirs(mask_path)
    tqdm_imap(copy_mask, list(enumerate(mask_files)))

    tmp_dataset = VTMDataset(tmp_dir)
    try:
        mesh = tsdf_fusion(tmp_dataset, mesh_options)
    except ValueError:  # ValueError is raised from the marching cubes function when tsdf_volume.min() > 0.
        mesh = None

    shutil.rmtree(tmp_dir)

    return mesh


def export_results(output_path):
    def read_results(path):
        with open(path, 'r') as f:
            results = json.load(f)
            pd_results_dict = defaultdict(lambda: defaultdict(float))

            for metric in results:
                for dataset in results[metric]:
                    for method in results[metric][dataset]:
                        if metric == 'rpe':
                            for sub_metric in results[metric][dataset][method]:
                                pd_results_dict[dataset, method][metric, sub_metric] = results[metric][dataset][method][
                                    sub_metric]
                        else:
                            pd_results_dict[dataset, method][metric, '-'] = results[metric][dataset][method]

            return pd.DataFrame.from_dict(pd_results_dict, orient='index')

    with pd.ExcelWriter(pjoin(output_path, 'results.xlsx')) as writer:
        trajectory_results_df = read_results(pjoin(output_path, 'trajectory', 'summary.json'))
        trajectory_results_df.to_excel(writer, sheet_name='TRAJECTORY_BY_DATASET')
        trajectory_results_df.groupby(level=1).mean().to_excel(writer, sheet_name='TRAJECTORY_BY_METHOD')

        pipeline_results_df = read_results(pjoin(output_path, 'pipeline', 'summary.json'))
        pipeline_results_df.to_excel(writer, sheet_name='PIPELINE_BY_DATASET')
        pipeline_results_df.groupby(level=1).mean().to_excel(writer, sheet_name='PIPELINE_BY_METHOD')


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

# TODO: Pull out each experiment type into own class.
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

        self.pipeline_configurations = [
            (self.gt_label, self.gt_options),
            (self.cm_label, self.cm_options),
            (self.dpt_label, self.dpt_options),
            (self.est_label, self.est_options),
        ]

        self.dataset_paths = self.generate_dataset_paths(output_path, dataset_names, self.pipeline_configurations)

        self.summaries_path = pjoin(self.output_path, 'summaries')
        self.latex_path = pjoin(self.output_path, 'latex')
        self.trajectory_outputs_path = pjoin(self.output_path, 'trajectory')
        self.tmp_path = pjoin(self.output_path, 'tmp')

        for path in (self.summaries_path, self.latex_path, self.tmp_path, self.trajectory_outputs_path):
            os.makedirs(path, exist_ok=True)

        self.pipeline_results_path = pjoin(self.summaries_path, 'pipeline.json')
        self.trajectory_results_path = pjoin(self.summaries_path, 'trajectory.json')
        self.kid_running_results_path = pjoin(self.summaries_path, 'kid_running.json')

        self.colmap_options = COLMAPOptions(quality='medium')
        self.webxr_options = WebXROptions(webxr_path=self.tmp_path)

    @staticmethod
    def generate_dataset_paths(output_path:str,
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

            is_valid_dataset = VTMDataset.is_valid_folder_structure(output_path)

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

                is_valid_dataset = VTMDataset.is_valid_folder_structure(self.dataset_paths[dataset_name][label])

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

                        runtime_breakdown[step][sub_step]['total_wall_time'].append(foreground_step_wall_times[sub_step])
                        runtime_breakdown[step][sub_step]['frame_time'].append(foreground_step_per_frame_wall_times[sub_step])

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
                    file_statistics_by_layer[layer]['uncompressed_file_size'].append(mesh_compression[layer]['uncompressed_file_size'])
                    file_statistics_by_layer[layer]['compressed_file_size'].append(mesh_compression[layer]['compressed_file_size'])
                    file_statistics_by_layer[layer]['data_saving'].append(mesh_compression[layer]['data_saving'])
                    file_statistics_by_layer[layer]['compression_ratio'].append(mesh_compression[layer]['compression_ratio'])

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
        latex_lines.append(f"\\textbf{{Total}} & & {Latex.to_mean_stddev(total_runtimes, formatter=Latex.format_timedelta)} & "
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
                           f"{Latex.to_mean_stddev(all_compressed_file_size, formatter=Latex.bytes_to_megabytes)} & "
                           f"{Latex.to_mean_stddev(all_uncompressed_file_size, formatter=Latex.bytes_to_megabytes)} & "
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
            dataset_gt = VTMDataset(pjoin(self.output_path, f"{dataset_name}_{self.gt_label}"))

            for label in (self.cm_label, self.est_label):
                dataset = VTMDataset(pjoin(self.output_path, f"{dataset_name}_{label}"))

                logging.info(f"Running trajectory comparison for dataset '{dataset_name}' and config '{label}'.")
                run_trajectory_comparisons(dataset, pred_trajectory=dataset.camera_trajectory,
                                           gt_trajectory=dataset_gt.camera_trajectory, dataset_name=dataset_name,
                                           pred_label=label, gt_label=self.gt_label,
                                           results_dict=trajectory_results,
                                           output_folder=self.trajectory_outputs_path,
                                           background_mesh_options=BackgroundMeshOptions(),
                                           frame_step=self.frame_step)

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
                    latex_lines.append(f" & {label} & {rpe_rotation:<9,.2f} & {rpe_translation:<9,.2f} & {ate:<9,.2f} \\\\")

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

        latex_lines.append(f" & all & {Latex.to_mean(all_data[self.cm_label]['rpe']['rotation'] + all_data[self.est_label]['rpe']['rotation']):<9} & "
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

        for dataset_name in self.dataset_names:
            dataset_gt = VTMDataset(pjoin(self.output_path, f"{dataset_name}_{self.gt_label}"))
            mesh_options = BackgroundMeshOptions()

            for label in (self.gt_label, self.cm_label, self.est_label):
                dataset = VTMDataset(pjoin(self.output_path, f"{dataset_name}_{label}"))

                logging.info(f"Running comparisons for dataset '{dataset_name}' and config '{label}'...")
                mesh_output_path = pjoin(recon_folder, dataset_name, label)
                os.makedirs(mesh_output_path, exist_ok=True)

                frame_set = list(range(0, self.num_frames, self.frame_step))

                if frame_set[-1] != self.num_frames - 1:
                    frame_set.append(self.num_frames - 1)

                logging.info('Creating ground truth mesh...')
                gt_mesh = tsdf_fusion(dataset_gt, mesh_options, frame_set=frame_set)
                gt_mesh.export(pjoin(mesh_output_path, f"{self.gt_label}.ply"))

                logging.info('Creating TSDFFusion mesh with estimated data...')
                pred_mesh = tsdf_fusion(dataset, mesh_options, frame_set=frame_set)
                pred_mesh.export(pjoin(mesh_output_path, f"{self.est_label}.ply"))

                # This is needed in case BundleFusion has already been run with the dataset.
                logging.info('Creating BundleFusion mesh with estimated data...')
                dataset.overwrite_ok = self.overwrite_ok

                bf_mesh = bundle_fusion(output_folder='bundle_fusion', dataset=dataset, options=mesh_options)
                bf_mesh.export(pjoin(mesh_output_path, 'bf.ply'))

                logging.info('Creating TSDFFusion mesh with COLMAP depth...')

                if label != self.gt_label and (mesh := tsdf_fusion_with_colmap(dataset, frame_set, mesh_options)):
                    mesh.export(pjoin(mesh_output_path, 'colmap_depth.ply'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_path', help='The path to save the experiment outputs and results to.',
                        required=True, type=str)
    parser.add_argument('--data_path', help='The path to the folder containing the datasets.',
                        required=True, type=str)
    parser.add_argument('--log_file', type=str, help='The path to save the logs to.', default='experiments.log')
    parser.add_argument('-y', dest='overwrite_ok', action='store_true', help='Whether to overwrite any old results.')
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
    # experiments.run_bundlefusion_experiments()
