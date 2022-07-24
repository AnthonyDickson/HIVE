import argparse
import json
import logging
import os.path
import shutil
import time
from collections import defaultdict
from os.path import join as pjoin
from typing import Tuple, Dict, Optional, List

import numpy as np
import pandas as pd
import torch.cuda
import trimesh

from video2mesh.dataset_adaptors import get_dataset
from video2mesh.fusion import tsdf_fusion, bundle_fusion
from video2mesh.geometry import Trajectory
from video2mesh.io import VTMDataset, temporary_trajectory, DatasetMetadata
from video2mesh.options import BackgroundMeshOptions, COLMAPOptions, ForegroundTrajectorySmoothingOptions, \
    PipelineOptions, StorageOptions, MeshDecimationOptions, MeshReconstructionMethod
from video2mesh.pipeline import Pipeline
from video2mesh.utils import setup_logger, tqdm_imap


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


def run_trajectory_comparisons(dataset, pred_trajectory: Trajectory, gt_trajectory: Trajectory,
                               dataset_name: str, pred_label: str, gt_label: str,
                               results_dict: dict, output_folder: str,
                               background_mesh_options: BackgroundMeshOptions):
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
        mesh = tsdf_fusion(dataset, background_mesh_options)
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

    tsdf_mesh_options = BackgroundMeshOptions(reconstruction_method=MeshReconstructionMethod.TSDFFusion,
                                              sdf_num_voxels=80000000, sdf_volume_size=10.0)
    rgbd_mesh_options = BackgroundMeshOptions(reconstruction_method=MeshReconstructionMethod.RGBD)
    mesh_decimation_options = MeshDecimationOptions(num_vertices_object=-1, num_vertices_background=-1)

    # TODO: Make the dataset list configurable.
    dataset_names = ['rgbd_dataset_freiburg1_desk',
                     'rgbd_dataset_freiburg3_walking_xyz',
                     'rgbd_dataset_freiburg3_sitting_xyz',
                     'rgbd_dataset_freiburg2_desk_with_person',
                     'rgbd_dataset_freiburg1_teddy',
                     'edwardsBay']

    datasets: Dict[Tuple[str, str], VTMDataset] = dict()
    gt_label = 'gt'

    # Run the pipeline on the datasets, record some basic performance stats and test foreground trajectory smoothing.
    logging.info("Running pipeline comparisons...")
    mesh_video_output_path = pjoin(output_path, 'pipeline')

    pipeline_stats = dict()

    fg_smoothing_settings = (
        ForegroundTrajectorySmoothingOptions(learning_rate=1e-5, num_epochs=10),
        ForegroundTrajectorySmoothingOptions(learning_rate=1e-5, num_epochs=25),
    )

    # TODO: Extract function.
    def run_pipeline_experiment(dataset_name: str, label: str, options: PipelineOptions,
                                dataset_path: Optional[str] = None):
        logging.info(f"Creating dataset for '{dataset_name}' and config '{label}'...")

        if dataset_path is None:
            dataset_path = pjoin(data_path, dataset_name)

        profiler = Profiler()

        with profiler:
            dataset = get_dataset(
                StorageOptions(base_path=dataset_path, overwrite_ok=overwrite_ok),
                colmap_options, options, output_path=pjoin(output_path, f"{dataset_name}_{label}")
            )

            logging.info(f"Running pipeline for dataset '{dataset_name}' and config '{label}'.")
            base_options = dict(
                options=PipelineOptions(num_frames, frame_step, log_file=log_file),
                storage_options=StorageOptions(output_path, overwrite_ok),
                decimation_options=mesh_decimation_options,
                static_mesh_options=tsdf_mesh_options,
                colmap_options=colmap_options
            )

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

        # TODO: Fix crash (code 137) when it gets to desk sequence + cm config.
        # # Pipeline using per-frame meshes.
        # logging.info(f"Running pipeline for dataset '{dataset_name}', config '{label}' "
        #              f"and rgbd_mesh config {rgbd_mesh_options}...")
        #
        # per_frame_mesh_options = {**base_options, "static_mesh_options": rgbd_mesh_options}
        # pipeline = Pipeline(**per_frame_mesh_options)
        # pipeline.run(dataset)
        #
        # export_path = pjoin(mesh_video_output_path, dataset_name, label, 'rgbd_mesh')
        # shutil.copytree(mesh_export_path, export_path, dirs_exist_ok=True)

        return dataset

    for dataset_name in dataset_names:
        for label, options in ((gt_label, gt_options), ('cm', cm_options), ('est', est_options)):
            datasets[label, dataset_name] = run_pipeline_experiment(dataset_name, label, options)

    # Ours vs NeRF based methods
    run_pipeline_experiment('kid_running', label='est', options=est_options,
                            dataset_path=pjoin(data_path, f"kid_running.mp4"))

    with open(pjoin(mesh_video_output_path, 'summary.json'), 'w') as f:
        json.dump(pipeline_stats, f)

    # Trajectory comparison
    logging.info("Running trajectory comparisons...")
    trajectory_results = dict()

    trajectory_results_path = pjoin(output_path, 'trajectory')
    os.makedirs(trajectory_results_path, exist_ok=True)

    for (label, dataset_name), dataset in datasets.items():
        logging.info(f"Running trajectory comparison for dataset '{dataset_name}' and config '{label}'.")
        run_trajectory_comparisons(dataset,
                                   pred_trajectory=dataset.camera_trajectory,
                                   gt_trajectory=datasets[gt_label, dataset_name].camera_trajectory,
                                   dataset_name=dataset_name,
                                   pred_label=label,
                                   gt_label=gt_label,
                                   results_dict=trajectory_results,
                                   output_folder=trajectory_results_path,
                                   background_mesh_options=tsdf_mesh_options)

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
        gt_mesh = tsdf_fusion(datasets[gt_label, dataset_name], tsdf_mesh_options, frame_set=frame_set)
        gt_mesh.export(pjoin(mesh_output_path, 'gt.ply'))

        logging.info('Creating TSDFFusion mesh with estimated data...')
        pred_mesh = tsdf_fusion(dataset, tsdf_mesh_options, frame_set=frame_set)
        pred_mesh.export(pjoin(mesh_output_path, 'pred.ply'))

        # This is needed in case BundleFusion has already been run with the dataset.
        logging.info('Creating BundleFusion mesh with estimated data...')
        dataset.overwrite_ok = overwrite_ok

        bf_mesh = bundle_fusion(output_folder='bundle_fusion', dataset=dataset, options=tsdf_mesh_options)
        bf_mesh.export(pjoin(mesh_output_path, 'bf.ply'))

        logging.info('Creating TSDFFusion mesh with COLMAP depth...')

        if label != gt_label and (mesh := tsdf_fusion_with_colmap(dataset, frame_set, tsdf_mesh_options)):
            mesh.export(pjoin(mesh_output_path, 'colmap_depth.ply'))

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
