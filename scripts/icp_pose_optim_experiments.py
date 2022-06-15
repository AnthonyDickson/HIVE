import argparse
import contextlib
import json
import os.path
import shutil
import warnings
from collections import defaultdict
from os.path import join as pjoin
from typing import Tuple, Optional

import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
import pandas as pd
from scipy.spatial.transform import Rotation
from tqdm import tqdm

from video2mesh.dataset_adaptors import TUMAdaptor
from video2mesh.fusion import tsdf_fusion, bundle_fusion
from video2mesh.geometry import pose_vec2mat, subtract_pose, pose_mat2vec, \
    get_identity_pose, add_pose, invert_trajectory
from video2mesh.io import VTMDataset
from video2mesh.options import StaticMeshOptions
from video2mesh.pose_optimisation import PoseOptimiser, FeatureExtractionOptions, OptimisationOptions, OptimisationStep
from video2mesh.utils import log


def setup(output_path: str, overwrite_ok: bool):
    if os.path.isdir(output_path):
        user_input = input(f"The output folder at {output_path} already exists, "
                           f"do you want to delete this folder before continuing? (y/n):")
        should_delete = user_input.lower() == 'y'

        if should_delete:
            shutil.rmtree(output_path)
        elif not overwrite_ok:
            raise RuntimeError(f"The output folder at {output_path} already exists. "
                               "Either change the output path or delete the existing folder.")
        else:
            user_input = input(f"The output folder at {output_path} already exists.\n"
                               f"Since `overwrite_ok` has been set to True, any existing results will be overwritten "
                               f"but the created datasets will be left as-is.\n"
                               f"This could lead to results from previous runs mixed up with the new results.\n"
                               f"Are you sure you want to continue? (y/n):")

            if user_input.lower() != 'y':
                exit(0)

    os.makedirs(output_path, exist_ok=overwrite_ok)


def calculate_ate(gt_trajectory, pred_trajectory) -> Tuple[np.ndarray, np.ndarray]:
    """
    Align two trajectories using the method of Horn (closed-form).
    Adapted from https://svncvpr.in.tum.de/cvpr-ros-pkg/trunk/rgbd_benchmark/rgbd_benchmark_tools/src/rgbd_benchmark_tools/evaluate_ate.py

    :param gt_trajectory: first trajectory (Nx3)
    :param pred_trajectory: second trajectory (Nx3)

    :return translational error per point (Nx1), aligned pred_trajectory (Nx3)
    """
    assert len(pred_trajectory.shape) == 2, \
        "trajectories must be a m x n array"
    assert pred_trajectory.shape[1] == 3, "trajectories must be a 3xN matrix."
    assert pred_trajectory.shape == gt_trajectory.shape, \
        "gt_trajectory and pred_trajectory must have the same shape"

    gt_trajectory = gt_trajectory.T
    pred_trajectory = pred_trajectory.T

    model_zero_centered = pred_trajectory - pred_trajectory.mean(axis=1).reshape((-1, 1))
    data_zero_centered = gt_trajectory - gt_trajectory.mean(axis=1).reshape((-1, 1))

    W = np.zeros((3, 3))

    for column in range(pred_trajectory.shape[1]):
        W += np.outer(model_zero_centered[:, column], data_zero_centered[:, column])

    U, d, Vh = np.linalg.linalg.svd(W.transpose())
    S = np.identity(3)

    if (np.linalg.det(U) * np.linalg.det(Vh) < 0):
        S[2, 2] = -1

    rotation = U * S * Vh
    translation = gt_trajectory.mean(axis=1) - np.matmul(rotation, pred_trajectory.mean(axis=1))
    # Calculations using translation assume it is a column vector, so we need to reshape here.
    translation = translation.reshape((3, 1))

    pred_trajectory_aligned = np.matmul(rotation, pred_trajectory) + translation
    alignment_error = pred_trajectory_aligned - gt_trajectory

    translational_error = np.sqrt(np.sum(np.square(alignment_error), axis=0))

    return translational_error.T, pred_trajectory_aligned.T


def visualise_ate(gt_trajectory, pred_trajectory, output_path):
    _, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.8, 4.8))

    def plot_trajectory(plot_axis, secondary_axis='y'):
        if secondary_axis == 'y':
            axis = 1
        elif secondary_axis == 'z':
            axis = 2
        else:
            raise RuntimeError(f"secondary_axis must be one of ('y', 'z').")

        plot_axis.plot(gt_trajectory[:, 0], gt_trajectory[:, axis], '-', color="black", label="ground truth")
        plot_axis.plot(pred_trajectory[:, 0], pred_trajectory[:, axis], '-', color="blue", label="estimated")
        plot_axis.legend()

        plot_axis.set_xlabel("x [m]")
        plot_axis.set_ylabel(f"{secondary_axis} [m]")
        plot_axis.set_title(f"Trajectory on X{secondary_axis.upper()} Plane")

    plot_trajectory(ax1, secondary_axis='y')
    plot_trajectory(ax2, secondary_axis='z')

    plt.tight_layout()
    plt.savefig(output_path, dpi=90)


def calculate_rpe(gt_trajectory, pred_trajectory):
    rotational_error = []
    translational_error = []

    num_frames = min(len(pred_trajectory), len(gt_trajectory))

    for i, j in zip(range(num_frames - 1), range(1, num_frames)):
        rel_pose_est = subtract_pose(pred_trajectory[i], pred_trajectory[j])
        rel_pose_gt = subtract_pose(gt_trajectory[i], gt_trajectory[j])
        rel_pose_error = subtract_pose(rel_pose_est, rel_pose_gt)

        distance = np.linalg.norm(rel_pose_error[4:])
        angle = np.arccos(min(1, max(-1, (np.trace(pose_vec2mat(rel_pose_error)[:3, :3]) - 1) / 2)))

        translational_error.append(distance)
        rotational_error.append(angle)

    return np.asarray(rotational_error), np.asarray(translational_error)


def create_point_cloud(dataset: VTMDataset, index: int):
    rgb = dataset.rgb_dataset[index]
    depth = dataset.depth_dataset[index]
    mask = dataset.mask_dataset[index] > 0
    depth[mask] = 0.0

    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
        o3d.geometry.RGBDImage.create_from_color_and_depth(
            color=o3d.geometry.Image(rgb),
            depth=o3d.geometry.Image(depth),
            depth_scale=1.0,
            depth_trunc=10.0,
            convert_rgb_to_intensity=False
        ),
        intrinsic=o3d.camera.PinholeCameraIntrinsic(dataset.frame_width, dataset.frame_height, dataset.fx, dataset.fy,
                                                    dataset.cx, dataset.cy)
    )

    return pcd


def preprocess_point_cloud(pcd, voxel_size):
    pcd_down = pcd.voxel_down_sample(voxel_size)

    radius_normal = voxel_size * 2
    pcd.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))

    pcd_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))

    radius_feature = voxel_size * 5
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))

    return pcd_down, pcd_fpfh


def register_with_ransac(source_down, target_down, source_fpfh,
                         target_fpfh, voxel_size):
    distance_threshold = voxel_size * 1.5

    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh, True,
        distance_threshold,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        3, [
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(
                0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(
                distance_threshold)
        ], o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999))

    return result


def icp(source, target):
    # colored pointcloud registration
    # From: http://www.open3d.org/docs/release/tutorial/pipelines/colored_pointcloud_registration.html
    # This is implementation of following paper
    # J. Park, Q.-Y. Zhou, V. Koltun,
    # Colored Point Cloud Registration Revisited, ICCV 2017
    voxel_radius = [0.04, 0.02, 0.01]
    max_iter = [60, 40, 20]
    times_skipped = 0

    # Use RANSAC to provide a good initial estimate.
    # If this step is skipped, this ICP algorithm sometimes fails on estimated depth.
    voxel_size = 0.05
    source_down, source_fpfh = preprocess_point_cloud(source, voxel_size)
    target_down, target_fpfh = preprocess_point_cloud(target, voxel_size)

    result_ransac = register_with_ransac(source_down, target_down,
                                         source_fpfh, target_fpfh,
                                         voxel_size)

    current_transformation = result_ransac.transformation

    for scale in range(3):
        iter = max_iter[scale]
        radius = voxel_radius[scale]

        source_down = source.voxel_down_sample(radius)
        target_down = target.voxel_down_sample(radius)

        source_down.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=radius * 2, max_nn=30))
        target_down.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=radius * 2, max_nn=30))

        try:
            result_icp = o3d.pipelines.registration.registration_colored_icp(
                source_down, target_down, radius, current_transformation,
                o3d.pipelines.registration.TransformationEstimationForColoredICP(),
                o3d.pipelines.registration.ICPConvergenceCriteria(relative_fitness=1e-6,
                                                                  relative_rmse=1e-6,
                                                                  max_iteration=iter))

            current_transformation = result_icp.transformation
        except RuntimeError as e:
            exception_message = str(e)

            if 'no correspondences' not in exception_message.lower():
                raise
            else:
                times_skipped += 1

    if times_skipped == len(voxel_radius) == len(max_iter):
        warnings.warn("Could not find correspondences between the source and target point clouds. "
                      "Using initial global alignment found with RANSAC.")

    return current_transformation.copy()


def get_icp_trajectory(dataset: VTMDataset):
    log(f"Aligning point clouds with ICP...")
    num_frames = dataset.num_frames
    relative_poses = []

    for frame_i, frame_j in tqdm(zip(range(num_frames - 1), range(1, num_frames)), total=num_frames):
        source = create_point_cloud(dataset, frame_i)
        target = create_point_cloud(dataset, frame_j)

        pose = icp(source, target)

        relative_poses.append(pose_mat2vec(pose))

    return np.asarray([get_identity_pose()] + relative_poses)


def merge_trajectory(relative_poses: np.ndarray) -> np.ndarray:
    # Assumes relative_poses[0] == get_identity_pose()
    merged_pose_data = []
    previous_pose = get_identity_pose()

    for pose in relative_poses:
        new_pose = add_pose(pose, previous_pose)

        merged_pose_data.append(new_pose)
        previous_pose = new_pose

    return np.asarray(merged_pose_data)


@contextlib.contextmanager
def temp_traj(dataset: VTMDataset, trajectory: np.ndarray):
    traj_backup = dataset.camera_trajectory.copy()

    try:
        dataset.camera_trajectory = trajectory

        yield
    finally:
        dataset.camera_trajectory = traj_backup


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
        icp_results_df = read_results(pjoin(output_path, 'icp', 'summary.json'))
        icp_results_df.to_excel(writer, sheet_name='ICP_BY_DATASET')
        icp_results_df.groupby(level=1).mean().to_excel(writer, sheet_name='ICP_BY_METHOD')

        ablation_results_df = read_results(pjoin(output_path, 'ablation', 'summary.json'))
        ablation_results_df.to_excel(writer, sheet_name='ABLATION_BY_DATASET')
        ablation_results_df.groupby(level=1).mean().to_excel(writer, sheet_name='ABLATION_BY_METHOD')


def main(output_path: str, data_path: str, random_seed: Optional[int] = None, overwrite_ok=False):
    setup(output_path=output_path, overwrite_ok=overwrite_ok)

    # TODO: Make the dataset list configurable.
    # TODO: Download any missing TUM datasets.
    datasets = ['rgbd_dataset_freiburg1_desk',
                'rgbd_dataset_freiburg3_walking_xyz',
                'rgbd_dataset_freiburg3_sitting_xyz']
    # ICP vs Our RGB-D Alignment
    num_frames = 150
    frame_step = 1
    icp_results = dict()

    static_mesh_options = StaticMeshOptions(sdf_num_voxels=80000000, sdf_volume_size=10.0)

    icp_results_path = pjoin(output_path, 'icp')
    os.makedirs(icp_results_path, exist_ok=True)

    def add_key(key, dataset_name, pred_label, results_dict):
        if key not in results_dict:
            results_dict[key] = dict()

        if dataset_name not in results_dict[key]:
            results_dict[key][dataset_name] = dict()

        if pred_label not in results_dict[key][dataset_name]:
            results_dict[key][dataset_name][pred_label] = dict()

    def run_trajectory_comparisons(dataset, pred_trajectory, gt_trajectory,
                                   pred_label: str, gt_label: str,
                                   results_dict: dict, output_folder: str):
        experiment_path = pjoin(output_folder, dataset_name, pred_label)
        os.makedirs(experiment_path, exist_ok=True)

        np.savetxt(pjoin(experiment_path, 'trajectory.txt'), pred_trajectory)

        ate, aligned_trajectory = calculate_ate(gt_trajectory[:, 4:], pred_trajectory[:, 4:])

        visualise_ate(gt_trajectory[:, 4:], aligned_trajectory,
                      output_path=pjoin(experiment_path, 'ate.png'))
        visualise_ate(gt_trajectory[:, 4:], pred_trajectory[:, 4:],
                      output_path=pjoin(experiment_path, 'trajectory_comparison.png'))

        # noinspection PyTypeChecker
        np.savetxt(pjoin(experiment_path, 'ate.txt'), ate)

        trajectory_error = gt_trajectory[:, 4:] - pred_trajectory[:, 4:]
        # noinspection PyTypeChecker
        np.savetxt(pjoin(experiment_path, 'trajectory_error.txt'), trajectory_error)

        with temp_traj(dataset, pred_trajectory):
            mesh = tsdf_fusion(dataset, static_mesh_options)
            mesh.export(pjoin(experiment_path, f"mesh.ply"))

        error_r, error_t = calculate_rpe(gt_trajectory, pred_trajectory)
        # noinspection PyTypeChecker
        np.savetxt(pjoin(experiment_path, f"rpe_r.txt"), error_r)
        # noinspection PyTypeChecker
        np.savetxt(pjoin(experiment_path, f"rpe_t.txt"), error_t)

        log(f"{dataset_name} - {pred_label.upper()} vs. {gt_label.upper()}:")
        log(f"\tATE: {np.mean(ate):.2f}m")
        log(f"\tMATE: {np.mean(np.abs(trajectory_error)):.2f}m")
        log(f"\tRPE (rot): {np.mean(np.rad2deg(error_r)):.2f}\N{DEGREE SIGN}")
        log(f"\tRPE (tra): {np.mean(error_t):.2f}m")

        add_key('rpe', dataset_name, pred_label, results_dict)
        add_key('ate', dataset_name, pred_label, results_dict)
        add_key('mate', dataset_name, pred_label, results_dict)

        results_dict['ate'][dataset_name][pred_label] = np.mean(ate)
        results_dict['mate'][dataset_name][pred_label] = np.mean(np.abs(trajectory_error))
        results_dict['rpe'][dataset_name][pred_label]['rotation'] = np.mean(np.rad2deg(error_r))
        results_dict['rpe'][dataset_name][pred_label]['translation'] = np.mean(error_t)

    def run_pose_optim_experiment(dataset: VTMDataset, options: OptimisationOptions, pred_label: str):
        cam_traj = PoseOptimiser(
            dataset,
            feature_extraction_options=FeatureExtractionOptions(min_features=20),
            optimisation_options=options,
            debug=False
        ).run()[0]

        run_trajectory_comparisons(dataset, cam_traj, dataset_gt.camera_trajectory,
                                   pred_label=pred_label, gt_label='gt',
                                   results_dict=ablation_results, output_folder=pjoin(output_path, 'ablation'))

    for dataset_name in datasets:
        dataset_gt = TUMAdaptor(
            base_path=pjoin(data_path, dataset_name),
            output_path=pjoin(output_path, f"{dataset_name}_gt"),
            num_frames=num_frames,
            frame_step=frame_step
        ).convert_from_ground_truth()

        run_trajectory_comparisons(dataset_gt, dataset_gt.camera_trajectory, dataset_gt.camera_trajectory,
                                   pred_label='gt', gt_label='gt',
                                   results_dict=icp_results, output_folder=icp_results_path)

        icp_trajectory = invert_trajectory(merge_trajectory(get_icp_trajectory(dataset_gt)))
        run_trajectory_comparisons(dataset_gt, icp_trajectory, dataset_gt.camera_trajectory, pred_label='icp',
                                   gt_label='gt', results_dict=icp_results, output_folder=icp_results_path)

        optimiser = PoseOptimiser(
            dataset_gt,
            feature_extraction_options=FeatureExtractionOptions(min_features=40),
            debug=True
        )
        optimised_trajectory, _, _ = optimiser.run(num_frames=dataset_gt.num_frames)
        run_trajectory_comparisons(dataset_gt, optimised_trajectory, dataset_gt.camera_trajectory, pred_label='ours',
                                   gt_label='gt', results_dict=icp_results, output_folder=icp_results_path)

        dataset_estimated = TUMAdaptor(
            base_path=pjoin(data_path, dataset_name),
            output_path=pjoin(output_path, f"{dataset_name}_est"),
            num_frames=num_frames,
            frame_step=frame_step
        ).convert_from_rgb()

        run_trajectory_comparisons(dataset_estimated, dataset_gt.camera_trajectory, dataset_gt.camera_trajectory,
                                   pred_label='gt_est', gt_label='gt',
                                   results_dict=icp_results, output_folder=icp_results_path)

        icp_trajectory_est = invert_trajectory(merge_trajectory(get_icp_trajectory(dataset_estimated)))
        run_trajectory_comparisons(dataset_estimated, icp_trajectory_est, dataset_gt.camera_trajectory,
                                   pred_label='icp_est', gt_label='colmap', results_dict=icp_results,
                                   output_folder=icp_results_path)

        run_trajectory_comparisons(dataset_estimated, dataset_estimated.camera_trajectory, dataset_gt.camera_trajectory,
                                   pred_label='ours_est', gt_label='colmap', results_dict=icp_results,
                                   output_folder=icp_results_path)

        # TODO: Real world video for ICP vs Ours?

    with open(pjoin(output_path, 'icp', 'summary.json'), 'w') as f:
        json.dump(icp_results, f)

    # Ours vs NeRF based methods
    # TODO: Run on same clips as the examples in the NeRF papers
    # TODO: Record runtime statistics (e.g., wall time, peak GPU memory usage)

    # Our RGB-D Alignment
    ablation_results = dict()

    prng = np.random.default_rng(random_seed)

    default_pipeline = (OptimisationStep.PairWise3D, OptimisationStep.Global3D)
    global_only_pipeline = (OptimisationStep.Global3D,)
    loop_pipeline = default_pipeline + default_pipeline
    pipeline_2d = (OptimisationStep.PairWise2D, OptimisationStep.Global2D)
    pipeline_fine_tuning = (OptimisationStep.PairWise3D, OptimisationStep.PairWise2D,
                            OptimisationStep.Global3D, OptimisationStep.Global2D)
    pipeline_fine_tuning_3d_2d = (OptimisationStep.PairWise3D, OptimisationStep.Global3D,
                                  OptimisationStep.PairWise2D, OptimisationStep.Global2D)

    for dataset_name in datasets:
        dataset_gt = TUMAdaptor(
            base_path=pjoin(data_path, dataset_name),
            output_path=pjoin(output_path, f"{dataset_name}_gt"),
            num_frames=num_frames,
            frame_step=frame_step
        ).convert_from_ground_truth()

        dataset_rand = VTMDataset(dataset_gt.base_path)
        dataset_rand.camera_trajectory = np.hstack((
            Rotation.random(len(dataset_gt.camera_trajectory), prng).as_quat(),
            prng.normal(size=(len(dataset_gt.camera_trajectory), 3))
        ))

        dataset_estimated = TUMAdaptor(
            base_path=pjoin(data_path, dataset_name),
            output_path=pjoin(output_path, f"{dataset_name}_est"),
            num_frames=num_frames,
            frame_step=frame_step
        ).convert_from_rgb()

        for dataset, pred_label_suffix in zip((dataset_gt, dataset_rand, dataset_estimated), ('_gt', '_rand', '_est')):
            run_pose_optim_experiment(dataset, OptimisationOptions(),
                                      pred_label=f"baseline{pred_label_suffix}")

            run_pose_optim_experiment(dataset, OptimisationOptions(steps=global_only_pipeline),
                                      pred_label=f"global_only{pred_label_suffix}")

            run_pose_optim_experiment(dataset, OptimisationOptions(steps=loop_pipeline),
                                      pred_label=f"loop{pred_label_suffix}")

            run_pose_optim_experiment(dataset, OptimisationOptions(steps=pipeline_2d),
                                      pred_label=f"2d_residuals{pred_label_suffix}")

            # Compare with and without a final fine-tuning step that uses the image-to-image projection residuals.
            run_pose_optim_experiment(dataset, OptimisationOptions(steps=pipeline_fine_tuning),
                                      pred_label=f"fine_tune{pred_label_suffix}")

            run_pose_optim_experiment(dataset, OptimisationOptions(steps=pipeline_fine_tuning_3d_2d),
                                      pred_label=f"fine_tune_3d_to_2d{pred_label_suffix}")

            # Compare optimising just the camera position vs the entire pose (position + orientation).
            run_pose_optim_experiment(dataset, OptimisationOptions(position_only=True),
                                      pred_label=f"pos_only{pred_label_suffix}")

    with open(pjoin(output_path, 'ablation', 'summary.json'), 'w') as f:
        json.dump(ablation_results, f)

    # TSDFFusion w/ Our RGB-D Alignment vs BundleFusion
    recon_folder = pjoin(output_path, 'reconstruction')

    for dataset_name in datasets:
        mesh_output_path = pjoin(recon_folder, dataset_name)
        os.makedirs(mesh_output_path, exist_ok=True)

        # TSDFFusion vs BundleFusion on GT data.
        dataset_gt = TUMAdaptor(
            base_path=pjoin(data_path, dataset_name),
            output_path=pjoin(output_path, f"{dataset_name}_gt"),
            num_frames=num_frames,
            frame_step=frame_step
        ).convert_from_ground_truth()
        # This is needed in case BundleFusion has already been run with the dataset.
        dataset_gt.overwrite_ok = overwrite_ok

        gt_mesh = tsdf_fusion(dataset_gt, static_mesh_options)
        gt_mesh.export(pjoin(mesh_output_path, 'gt.ply'))

        tsdf_fusion_mesh = tsdf_fusion(dataset_gt, static_mesh_options)
        tsdf_fusion_mesh.export(pjoin(mesh_output_path, 'tsdf.ply'))

        bf_mesh = bundle_fusion(output_folder='bundle_fusion', dataset=dataset_gt, options=static_mesh_options)
        bf_mesh.export(pjoin(mesh_output_path, 'bf.ply'))

        # TSDFFusion vs TSDFFusion w/ Our Alignment vs BundleFusion on Estimated Data.
        dataset_estimated = TUMAdaptor(
            base_path=pjoin(data_path, dataset_name),
            output_path=pjoin(output_path, f"{dataset_name}_est"),
            num_frames=num_frames,
            frame_step=frame_step
        ).convert_from_rgb()
        # This is needed in case BundleFusion has already been run with the dataset.
        dataset_estimated.overwrite_ok = overwrite_ok

        with temp_traj(dataset_estimated, dataset_gt.camera_trajectory):
            gt_mesh_est = tsdf_fusion(dataset_estimated, static_mesh_options)
            gt_mesh_est.export(pjoin(mesh_output_path, 'gt_est.ply'))

        tsdf_fusion_mesh_est = tsdf_fusion(dataset_estimated, static_mesh_options)
        tsdf_fusion_mesh_est.export(pjoin(mesh_output_path, 'tsdf_est.ply'))

        bf_mesh_est = bundle_fusion(output_folder='bundle_fusion', dataset=dataset_estimated,
                                    options=static_mesh_options)
        bf_mesh_est.export(pjoin(mesh_output_path, 'bf_est.ply'))

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

    main(output_path=args.output_path, data_path=args.data_path, random_seed=args.random_seed,
         overwrite_ok=args.overwrite_ok)
