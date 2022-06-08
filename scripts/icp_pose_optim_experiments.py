import argparse
import contextlib
import json
import os.path
import shutil
import warnings
from os.path import join as pjoin
from typing import Tuple, Optional

import matplotlib

from video2mesh.fusion import tsdf_fusion, bundle_fusion
from video2mesh.options import StaticMeshOptions, PipelineOptions, StorageOptions
from video2mesh.utils import log, tqdm_imap

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d

from video2mesh.dataset_adaptors import TUMAdaptor
from video2mesh.geometry import pose_vec2mat, subtract_pose, pose_mat2vec, \
    get_identity_pose, add_pose
from video2mesh.io import VTMDataset
from video2mesh.pipeline import Pipeline
from video2mesh.pose_optimisation import PoseOptimiser, FeatureExtractionOptions, OptimisationOptions


def setup(output_path: str, overwrite_ok: bool):
    if os.path.isdir(output_path):
        if not overwrite_ok:
            user_input = input(f"The output folder at {output_path} already exists, "
                               f"do you want to delete this folder before continuing? (y/n):")
            should_delete = user_input == 'y'
        else:
            should_delete = True

        if should_delete:
            shutil.rmtree(output_path)
        else:
            raise RuntimeError(f"The output folder at {output_path} already exists. "
                               "Either change the output path or delete the existing folder.")

    os.makedirs(output_path)


def align_for_ate(gt_trajectory, pred_trajectory) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Align two trajectories using the method of Horn (closed-form).
    Adapted from https://svncvpr.in.tum.de/cvpr-ros-pkg/trunk/rgbd_benchmark/rgbd_benchmark_tools/src/rgbd_benchmark_tools/evaluate_ate.py

    :param gt_trajectory: first trajectory (3xn)
    :param pred_trajectory: second trajectory (3xn)

    :return rot -- rotation matrix (3x3), translation -- translation vector (3x1),
        trans_error -- translational error per point (1xn)
    """
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

    return rotation, translation, translational_error


def calculate_ate(gt_trajectory, pred_trajectory) -> Tuple[np.ndarray, np.ndarray]:
    # 3xN matrix
    rotation, translation, translational_error = align_for_ate(gt_trajectory[:, 4:].T, pred_trajectory[:, 4:].T)

    aligned_trajectory = (np.matmul(rotation, pred_trajectory[:, 4:].T) + translation).T

    return translational_error, aligned_trajectory


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


def similarity_transform(from_points, to_points):
    # https://gist.github.com/dboyliao/f7f862172ed811032ba7cc368701b1e8
    assert len(from_points.shape) == 2, \
        "from_points must be a m x n array"
    assert from_points.shape == to_points.shape, \
        "from_points and to_points must have the same shape"

    N, m = from_points.shape

    mean_from = from_points.mean(axis=0)
    mean_to = to_points.mean(axis=0)

    delta_from = from_points - mean_from  # N x m
    delta_to = to_points - mean_to  # N x m

    sigma_from = (delta_from * delta_from).sum(axis=1).mean()
    sigma_to = (delta_to * delta_to).sum(axis=1).mean()

    cov_matrix = delta_to.T.dot(delta_from) / N

    U, d, V_t = np.linalg.svd(cov_matrix, full_matrices=True)
    cov_rank = np.linalg.matrix_rank(cov_matrix)
    S = np.eye(m)

    if cov_rank >= m - 1 and np.linalg.det(cov_matrix) < 0:
        S[m - 1, m - 1] = -1
    elif cov_rank < m - 1:
        raise ValueError("colinearility detected in covariance matrix:\n{}".format(cov_matrix))

    R = U.dot(S).dot(V_t)
    c = (d * S.diagonal()).sum() / sigma_from
    t = mean_to - c * R.dot(mean_from)

    return c * R, t


def create_point_cloud_o3d(dataset: VTMDataset, index: int):
    rgb = dataset.rgb_dataset[index]
    depth = dataset.depth_dataset[index]
    mask = dataset.mask_dataset[index] > 0
    depth[mask] = 0.0

    pose = pose_vec2mat(dataset.camera_trajectory[index])

    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
        o3d.geometry.RGBDImage.create_from_color_and_depth(
            color=o3d.geometry.Image(rgb),
            depth=o3d.geometry.Image(depth),
            depth_scale=1.0,
            depth_trunc=10.0,
            convert_rgb_to_intensity=False
        ),
        intrinsic=o3d.camera.PinholeCameraIntrinsic(dataset.frame_width, dataset.frame_height, dataset.fx, dataset.fy,
                                                    dataset.cx, dataset.cy),
        extrinsic=pose
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


def icp_o3d(source, target):
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
    def align(frame_indices):
        frame_i, frame_j = frame_indices

        source = create_point_cloud_o3d(dataset, frame_i)
        target = create_point_cloud_o3d(dataset, frame_j)

        pose = icp_o3d(source, target)

        return pose_mat2vec(pose)

    num_frames = dataset.num_frames

    log(f"Aligning point clouds with ICP...")
    relative_poses = tqdm_imap(align, list(zip(range(num_frames - 1), range(1, num_frames))))

    return np.asarray([get_identity_pose()] + relative_poses)


def merge_trajectory(relative_poses: np.ndarray) -> np.ndarray:
    merged_pose_data = [get_identity_pose()]
    previous_pose = merged_pose_data[0]

    num_frames = len(relative_poses)

    for i, j in zip(range(num_frames - 1), range(1, num_frames)):
        pose_i, pose_j = relative_poses[[i, j]]
        j_rel_to_i = subtract_pose(pose_i, pose_j)
        j_rel_to_world_origin = add_pose(previous_pose, j_rel_to_i)

        merged_pose_data.append(j_rel_to_world_origin)
        previous_pose = j_rel_to_world_origin

    return np.asarray(merged_pose_data)


@contextlib.contextmanager
def temp_traj(dataset: VTMDataset, trajectory: np.ndarray):
    traj_backup = dataset.camera_trajectory.copy()

    try:
        dataset.camera_trajectory = trajectory

        yield
    finally:
        dataset.camera_trajectory = traj_backup


def main(output_path: str, data_path: str, random_seed: Optional[int] = None, overwrite_ok=False):
    setup(output_path=output_path, overwrite_ok=overwrite_ok)

    # TODO: Make the dataset list configurable.
    # TODO: Download any missing TUM datasets.
    datasets = ['rgbd_dataset_freiburg3_walking_xyz',
                'rgbd_dataset_freiburg3_sitting_xyz',
                'rgbd_dataset_freiburg1_desk']
    # ICP vs Our RGB-D Alignment
    num_frames = 150
    frame_step = 1
    icp_results = dict()

    static_mesh_options = StaticMeshOptions(sdf_num_voxels=80000000, sdf_volume_size=10.0)

    icp_results_path = pjoin(output_path, 'icp')
    os.makedirs(icp_results_path, exist_ok=True)

    def run_trajectory_comparisons(dataset, pred_trajectory, gt_trajectory, pred_label: str, gt_label: str):
        experiment_path = pjoin(icp_results_path, dataset_name, pred_label)
        os.makedirs(experiment_path, exist_ok=True)

        translational_error, aligned_trajectory = calculate_ate(gt_trajectory, pred_trajectory)
        visualise_ate(gt_trajectory[:, 4:], aligned_trajectory,
                      output_path=pjoin(experiment_path, 'ate.png'))
        # noinspection PyTypeChecker
        np.savetxt(pjoin(experiment_path, 'ate.txt'), translational_error)
        log(f"{dataset_name} - {pred_label.upper()} vs. {gt_label.upper()}: ATE: {np.mean(translational_error):.2f}m")

        if 'ate' not in icp_results:
            icp_results['ate'] = dict()

        if dataset_name not in icp_results['ate']:
            icp_results['ate'][dataset_name] = dict()

        icp_results['ate'][dataset_name][pred_label] = np.mean(translational_error)

        with temp_traj(dataset, pred_trajectory):
            mesh = tsdf_fusion(dataset, static_mesh_options)
            mesh.export(pjoin(experiment_path, f"mesh.ply"))

        error_r, error_t = calculate_rpe(gt_trajectory, pred_trajectory)
        # noinspection PyTypeChecker
        np.savetxt(pjoin(experiment_path, f"rpe_r.txt"), error_r)
        # noinspection PyTypeChecker
        np.savetxt(pjoin(experiment_path, f"rpe_t.txt"), error_t)

        if 'rpe' not in icp_results:
            icp_results['rpe'] = dict()

        if dataset_name not in icp_results['rpe']:
            icp_results['rpe'][dataset_name] = dict()

        icp_results['rpe'][dataset_name]['rotation'] = np.mean(error_r)
        icp_results['rpe'][dataset_name]['translation'] = np.mean(error_t)

    for dataset_name in datasets:
        dataset_gt = TUMAdaptor(
            base_path=pjoin(data_path, dataset_name),
            output_path=pjoin(output_path, f"{dataset_name}_gt"),
            num_frames=num_frames,
            frame_step=frame_step
        ).convert_from_ground_truth()

        gt_output_path = pjoin(icp_results_path, dataset_name, 'gt')
        os.makedirs(gt_output_path, exist_ok=True)

        mesh = tsdf_fusion(dataset_gt, static_mesh_options)
        mesh.export(pjoin(gt_output_path, 'mesh.ply'))

        icp_trajectory = merge_trajectory(get_icp_trajectory(dataset_gt))
        run_trajectory_comparisons(dataset_gt, icp_trajectory, dataset_gt.camera_trajectory, pred_label='icp',
                                   gt_label='gt')

        optimiser = PoseOptimiser(
            dataset_gt,
            feature_extraction_options=FeatureExtractionOptions(min_features=40),
            debug=True
        )
        optimised_trajectory, _, _ = optimiser.run(num_frames=dataset_gt.num_frames)
        run_trajectory_comparisons(dataset_gt, optimised_trajectory, dataset_gt.camera_trajectory, pred_label='ours',
                                   gt_label='gt')

        dataset_estimated = TUMAdaptor(
            base_path=pjoin(data_path, dataset_name),
            output_path=pjoin(output_path, f"{dataset_name}_est"),
            num_frames=num_frames,
            frame_step=frame_step
        ).convert_from_rgb()

        with temp_traj(dataset_estimated, dataset_gt.camera_trajectory):
            gt_est_output_path = pjoin(icp_results_path, dataset_name, 'gt_est')
            os.makedirs(gt_est_output_path, exist_ok=True)

            mesh = tsdf_fusion(dataset_estimated, static_mesh_options)
            mesh.export(pjoin(gt_est_output_path, 'mesh.ply'))

        icp_trajectory_est = merge_trajectory(get_icp_trajectory(dataset_estimated))
        run_trajectory_comparisons(dataset_estimated, icp_trajectory_est, dataset_gt.camera_trajectory,
                                   pred_label='icp_est', gt_label='gt')

        run_trajectory_comparisons(dataset_estimated, dataset_estimated.camera_trajectory, dataset_gt.camera_trajectory,
                                   pred_label='ours_est', gt_label='gt')

        # TODO: Real world video for ICP vs Ours?

    with open(pjoin(output_path, 'icp', 'summary.json'), 'w') as f:
        json.dump(icp_results, f)

    # Ours vs NeRF based methods
    # TODO: Run on same clips as the examples in the NeRF papers
    # TODO: Record runtime statistics (e.g., wall time, peak GPU memory usage)

    # Our RGB-D Alignment
    ablation_results = dict()

    def run_pose_optim_experiment(dataset: VTMDataset, options: OptimisationOptions, pred_label: str):
        experiment_output_path = pjoin(output_path, 'ablation', dataset_name, pred_label)
        os.makedirs(experiment_output_path, exist_ok=True)

        cam_traj = PoseOptimiser(
            dataset,
            feature_extraction_options=FeatureExtractionOptions(min_features=40),
            optimisation_options=options,
            debug=False
        ).run()[0]

        np.savetxt(pjoin(experiment_output_path, "camera_trajectory.txt"), cam_traj)

        rpe_r, rpe_t = calculate_rpe(dataset.camera_trajectory, cam_traj)
        # noinspection PyTypeChecker
        np.savetxt(pjoin(experiment_output_path, "rpe_r.txt"), rpe_r)
        # noinspection PyTypeChecker
        np.savetxt(pjoin(experiment_output_path, "rpe_t.txt"), rpe_t)

        ate, _ = calculate_ate(dataset.camera_trajectory, cam_traj)
        # noinspection PyTypeChecker
        np.savetxt(pjoin(experiment_output_path, "ate.txt"), ate)

        with temp_traj(dataset, cam_traj):
            mesh = tsdf_fusion(dataset, static_mesh_options)
            mesh.export(pjoin(experiment_output_path, "mesh.ply"))

        def add_key(key):
            if key not in ablation_results:
                ablation_results[key] = dict()

            if dataset_name not in ablation_results[key]:
                ablation_results[key][dataset_name] = dict()

        add_key('rpe')
        add_key('ate')

        ablation_results['rpe'][dataset_name]['rotation'] = np.mean(rpe_r)
        ablation_results['rpe'][dataset_name]['translation'] = np.mean(rpe_t)
        ablation_results['ate'][dataset_name] = np.mean(ate)

    for dataset_name in datasets:
        rgbd_alignment_experiment_path = pjoin(output_path, 'ablation', dataset_name)
        os.makedirs(rgbd_alignment_experiment_path, exist_ok=True)

        # Compare optimising just the camera position vs the entire pose (position + orientation).
        dataset_gt = TUMAdaptor(
            base_path=pjoin(data_path, datasets[0]),
            output_path=pjoin(output_path, f"{datasets[0]}_gt"),
            num_frames=num_frames,
            frame_step=frame_step
        ).convert_from_ground_truth()

        dataset_estimated = TUMAdaptor(
            base_path=pjoin(data_path, datasets[0]),
            output_path=pjoin(output_path, f"{datasets[0]}_est"),
            num_frames=num_frames,
            frame_step=frame_step
        ).convert_from_rgb()

        run_pose_optim_experiment(dataset_gt, OptimisationOptions(position_only=True), pred_label='pos_only')
        run_pose_optim_experiment(dataset_gt, OptimisationOptions(position_only=False), pred_label='pos_rot')

        run_pose_optim_experiment(dataset_estimated, OptimisationOptions(position_only=True), pred_label='pos_only_est')
        run_pose_optim_experiment(dataset_estimated, OptimisationOptions(position_only=False), pred_label='pos_rot_est')

        # Compare with and without a final fine-tuning step that uses the image-to-image projection residuals.
        run_pose_optim_experiment(dataset_gt, OptimisationOptions(fine_tune=True), pred_label='fine_tune')
        run_pose_optim_experiment(dataset_gt, OptimisationOptions(fine_tune=False), pred_label='no_fine_tune')

        run_pose_optim_experiment(dataset_estimated, OptimisationOptions(fine_tune=True), pred_label='fine_tune_est')
        run_pose_optim_experiment(dataset_estimated, OptimisationOptions(fine_tune=False),
                                  pred_label='no_fine_tune_est')

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

        with temp_traj(dataset_estimated, dataset_gt.camera_trajectory):
            gt_mesh_est = tsdf_fusion(dataset_estimated, static_mesh_options)
            gt_mesh_est.export(pjoin(mesh_output_path, 'gt_est.ply'))

        tsdf_fusion_mesh_est = tsdf_fusion(dataset_estimated, static_mesh_options)
        tsdf_fusion_mesh_est.export(pjoin(mesh_output_path, 'tsdf_est.ply'))
        bf_mesh_est = bundle_fusion(output_folder='bundle_fusion', dataset=dataset_estimated,
                                    options=static_mesh_options)
        bf_mesh_est.export(pjoin(mesh_output_path, 'bf_est.ply'))

    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_path', help='The path to save the experiment outputs and results to.',
                        required=True, type=str)
    parser.add_argument('--data_path', help='The path to the folder containing the datasets.',
                        required=True, type=str)
    parser.add_argument('--random_seed', help='(optional) The seed to use for anything dealing with RNGs. '
                                              'If None, the random seed is not modified in any way.',
                        required=False, default=None, type=int)
    parser.add_argument('-y', dest='overwrite_ok', action='store_true', help='Whether to overwrite any1 old results.')
    args = parser.parse_args()

    main(output_path=args.output_path, data_path=args.data_path, random_seed=args.random_seed,
         overwrite_ok=args.overwrite_ok)
