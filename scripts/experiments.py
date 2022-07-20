import argparse
import contextlib
import datetime
import json
import logging
import os.path
import shutil
import time
import warnings
from collections import defaultdict
from os.path import join as pjoin
from typing import Tuple, Optional

import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
import pandas as pd
import torch.cuda
from scipy.spatial.transform import Rotation
from tqdm import tqdm

from video2mesh.dataset_adaptors import TUMAdaptor, get_dataset
from video2mesh.fusion import tsdf_fusion, bundle_fusion
from video2mesh.geometry import pose_vec2mat, subtract_pose, pose_mat2vec, \
    get_identity_pose, add_pose, Trajectory
from video2mesh.io import VTMDataset, temporary_trajectory
from video2mesh.options import BackgroundMeshOptions, COLMAPOptions, ForegroundTrajectorySmoothingOptions, \
    PipelineOptions, StorageOptions
from video2mesh.pipeline import Pipeline
from video2mesh.utils import setup_logger


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


def main(output_path: str, data_path: str, overwrite_ok=False):
    logging.info("Starting experiments...")
    logging.debug(str(dict(output_path=output_path, data_path=data_path, overwrite_ok=overwrite_ok)))

    logging.info("Setting up folders...")
    setup(output_path=output_path, overwrite_ok=overwrite_ok)

    log_file = 'experiments.log'
    setup_logger(log_file)

    logging.info("Creating datasets...")
    # TODO: Download any missing TUM datasets.
    colmap_options = COLMAPOptions(quality='medium', dense=True)

    num_frames = 300
    frame_step = 10

    gt_options = PipelineOptions(num_frames=num_frames, frame_step=frame_step, estimate_pose=False,
                                 estimate_depth=False)
    cm_options = PipelineOptions(num_frames=num_frames, frame_step=frame_step, estimate_pose=True, estimate_depth=False)
    est_options = PipelineOptions(num_frames=num_frames, frame_step=frame_step, estimate_pose=True, estimate_depth=True)

    static_mesh_options = BackgroundMeshOptions(sdf_num_voxels=80000000, sdf_volume_size=10.0)

    # TODO: Make the dataset list configurable.
    dataset_names = ['rgbd_dataset_freiburg1_desk',
                     'rgbd_dataset_freiburg3_walking_xyz',
                     'rgbd_dataset_freiburg3_sitting_xyz']

    datasets: Dict[Tuple[str, str], VTMDataset] = dict()
    gt_label = 'gt'

    for dataset_name in dataset_names:
        for label, options in ((gt_label, gt_options), ('cm', cm_options), ('est', est_options)):
            logging.info(f"Creating dataset for '{dataset_name}' and config '{label}'...")

            datasets[label, dataset_name] = get_dataset(
                StorageOptions(base_path=pjoin(data_path, dataset_name), overwrite_ok=overwrite_ok),
                colmap_options, options, output_path=pjoin(output_path, f"{dataset_name}_{label}")
            )

    # Trajectory comparison
    logging.info("Running trajectory comparisons...")
    trajectory_results = dict()

    trajectory_results_path = pjoin(output_path, 'trajectory')
    os.makedirs(trajectory_results_path, exist_ok=True)

    def add_key(key, dataset_name, pred_label, results_dict):
        if key not in results_dict:
            results_dict[key] = dict()

        if dataset_name not in results_dict[key]:
            results_dict[key][dataset_name] = dict()

        if pred_label not in results_dict[key][dataset_name]:
            results_dict[key][dataset_name][pred_label] = dict()

    def run_trajectory_comparisons(dataset, pred_trajectory: Trajectory, gt_trajectory: Trajectory,
                                   pred_label: str, gt_label: str,
                                   results_dict: dict, output_folder: str):
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
        np.savetxt(pjoin(experiment_path, f"rpe_r.txt"), error_r)
        # noinspection PyTypeChecker
        np.savetxt(pjoin(experiment_path, f"rpe_t.txt"), error_t)

        def rmse(x: np.ndarray) -> float:
            return np.sqrt(np.mean(np.square(x)))

        logging.info(f"{dataset_name} - {pred_label.upper()} vs. {gt_label.upper()}:")
        logging.info(f"\tATE: {rmse(ate):.2f}m")
        logging.info(f"\tRPE (rot): {rmse(np.rad2deg(error_r)):.2f}\N{DEGREE SIGN}")
        logging.info(f"\tRPE (tra): {rmse(error_t):.2f}m")

        add_key('rpe', dataset_name, pred_label, results_dict)
        add_key('ate', dataset_name, pred_label, results_dict)

        results_dict['ate'][dataset_name][pred_label] = rmse(ate)
        results_dict['rpe'][dataset_name][pred_label]['rotation'] = rmse(np.rad2deg(error_r))
        results_dict['rpe'][dataset_name][pred_label]['translation'] = rmse(error_t)

        with temporary_trajectory(dataset, pred_trajectory):
            mesh = tsdf_fusion(dataset, static_mesh_options)
            mesh.export(pjoin(experiment_path, f"mesh.ply"))

    # Ours vs NeRF based methods
    # TODO: Run on same clips as the examples in the NeRF papers
    # TODO: Record runtime statistics (e.g., wall time, peak GPU memory usage)

    for (label, dataset_name), dataset in datasets.items():
        logging.info(f"Running trajectory comparison for dataset '{dataset_name}' and config '{label}'.")
        run_trajectory_comparisons(dataset,
                                   pred_trajectory=dataset.camera_trajectory,
                                   gt_trajectory=datasets[gt_label, dataset_name].camera_trajectory,
                                   pred_label=label,
                                   gt_label=gt_label,
                                   results_dict=trajectory_results,
                                   output_folder=trajectory_results_path)

    with open(pjoin(trajectory_results_path, 'summary.json'), 'w') as f:
        json.dump(trajectory_results, f)

    # Scaled COLMAP + TSDFFusion vs BundleFusion
    logging.info("Running reconstruction comparisons...")
    recon_folder = pjoin(output_path, 'reconstruction')

    for (label, dataset_name), dataset in datasets.items():
        logging.info(f"Running comparisons for dataset '{dataset_name}' and config '{label}'...")
        mesh_output_path = pjoin(recon_folder, dataset_name, label)
        os.makedirs(mesh_output_path, exist_ok=True)

        frame_set = list(range(0, num_frames, frame_step))

        if frame_set[-1] != num_frames - 1:
            frame_set.append(num_frames - 1)

        logging.info('Creating ground truth mesh...')
        gt_mesh = tsdf_fusion(datasets[gt_label, dataset_name], static_mesh_options, frame_set=frame_set)
        gt_mesh.export(pjoin(mesh_output_path, 'gt.ply'))

        logging.info('Creating TSDFFusion mesh with estimated data...')
        pred_mesh = tsdf_fusion(dataset, static_mesh_options, frame_set=frame_set)
        pred_mesh.export(pjoin(mesh_output_path, 'pred.ply'))

        # This is needed in case BundleFusion has already been run with the dataset.
        logging.info('Creating BundleFusion mesh with estimated data...')
        dataset.overwrite_ok = overwrite_ok

        bf_mesh = bundle_fusion(output_folder='bundle_fusion', dataset=dataset, options=static_mesh_options)
        bf_mesh.export(pjoin(mesh_output_path, 'bf.ply'))

    # Run the pipeline on the datasets, record some basic performance stats and test foreground trajectory smoothing.
    logging.info("Running pipeline comparisons...")
    mesh_video_output_path = pjoin(output_path, 'pipeline')

    pipeline_stats = dict()

    class Profiler:
        def __init__(self):
            self.start = time.time()
            self.end = time.time()

            if torch.cuda.is_available():
                self.peak_gpu_memory_usage = torch.cuda.max_memory_allocated()
            else:
                self.peak_gpu_memory_usage = None

        def __enter__(self):
            self.start = time.time()

            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()

        def __exit__(self, exc_type, exc_val, exc_tb):
            self.end = time.time()

            if torch.cuda.is_available():
                self.peak_gpu_memory_usage = torch.cuda.max_memory_allocated()

        @property
        def elapsed(self) -> float:
            """Get the runtime in seconds."""
            if self.end > self.start:  # this means the time has probably been run properly.
                return self.end - self.start
            else:
                return time.time() - self.start

    fg_smoothing_settings = (
        ForegroundTrajectorySmoothingOptions(learning_rate=1e-5, num_epochs=10),
        ForegroundTrajectorySmoothingOptions(learning_rate=1e-5, num_epochs=25),
    )

    for (label, dataset_name), dataset in datasets.items():
        logging.info(f"Running pipeline for dataset '{dataset_name}' and config '{label}'.")
        base_options = dict(
            options=PipelineOptions(num_frames, frame_step, log_file=log_file),
            storage_options=StorageOptions(output_path, overwrite_ok),
            static_mesh_options=static_mesh_options,
            colmap_options=colmap_options
        )

        # TODO: Need to run dataset adaptor for profiler to capture GPU RAM usage.
        profiler = Profiler()

        with profiler:
            pipeline = Pipeline(**base_options)
            pipeline.run(dataset)

        export_path = pjoin(mesh_video_output_path, dataset_name, label, 'no_smoothing')
        mesh_export_path = pjoin(dataset.base_path, Pipeline.mesh_folder)
        shutil.copytree(mesh_export_path, export_path, dirs_exist_ok=True)

        add_key('runtime', dataset_name, label, pipeline_stats)
        add_key('peak_gpu_memory_usage', dataset_name, label, pipeline_stats)

        pipeline_stats['runtime'][dataset_name][label] = profiler.elapsed
        pipeline_stats['peak_gpu_memory_usage'][dataset_name][label] = profiler.peak_gpu_memory_usage

        for fg_smoothing_config in fg_smoothing_settings:
            logging.info(f"Running pipeline for dataset '{dataset_name}', config '{label}' "
                         f"and foreground smoothing config {fg_smoothing_config}")

            pipeline = Pipeline(**base_options, fts_options=fg_smoothing_config)
            pipeline.run(dataset)

            export_path = pjoin(
                mesh_video_output_path, dataset_name, label,
                f"smoothing_lr={fg_smoothing_config.learning_rate}_epochs={fg_smoothing_config.num_epochs}"
            )
            shutil.copytree(mesh_export_path, export_path, dirs_exist_ok=True)

    with open(pjoin(mesh_video_output_path, 'summary.json'), 'w') as f:
        json.dump(pipeline_stats, f)

    export_results(output_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_path', help='The path to save the experiment outputs and results to.',
                        required=True, type=str)
    parser.add_argument('--data_path', help='The path to the folder containing the datasets.',
                        required=True, type=str)
    parser.add_argument('--random_seed', help='(optional) The seed to use for anything dealing with RNGs. '
                                              'If None, the random seed is not modified in any way.',
                        required=False, default=None, type=int)
    parser.add_argument('-y', dest='overwrite_ok', action='store_true', help='Whether to overwrite any old results.')
    args = parser.parse_args()

    main(output_path=args.output_path, data_path=args.data_path, overwrite_ok=args.overwrite_ok)
