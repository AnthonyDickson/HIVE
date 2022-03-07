import os.path

import argparse
import datetime
import numpy as np
import trimesh
from scipy.spatial.transform import Rotation

from Video2mesh.geometry import pose_vec2mat, point_cloud_from_depth, point_cloud_from_rgbd
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

    frame = dataset.rgb_dataset[0]
    depth_map = dataset.depth_dataset[0]
    pose = dataset.camera_trajectory[0]
    mask_encoded = dataset.mask_dataset[0]
    binary_mask = mask_encoded == 0

    rgb = np.ascontiguousarray(frame[:, :, :3])
    pose_matrix = pose_vec2mat(pose)
    pose_matrix = np.linalg.inv(pose_matrix)
    R = pose_matrix[:3, :3]
    t = pose_matrix[:3, 3:]

    colour_points, points3d = point_cloud_from_rgbd(rgb, depth_map, binary_mask, dataset.camera_matrix, R, t)
    points3d_homogenous = np.ones(shape=(points3d.shape[0], 4))
    points3d_homogenous[:, :3] = points3d

    rot_180 = Rotation.from_euler('xyz', [0, 0, 0], degrees=True).as_matrix()
    centering_transform = np.eye(4, dtype=np.float32)
    centering_transform[:3, :3] = rot_180 @ R.T
    centering_transform[:3, 3:] = -(R.T @ t)

    points3d = (centering_transform @ points3d_homogenous.T).T[:, :3]

    pcd = trimesh.PointCloud(points3d, colour_points)

    bf_mesh_path = os.path.join(dataset.base_path, 'bundle_fusion', 'mesh.ply')
    bf_mesh: trimesh.Trimesh = trimesh.load(bf_mesh_path)

    rotation_transform = np.eye(4)
    rotation = Rotation.from_euler('xyz', [180, 0, 0], degrees=True).as_matrix()
    rotation_transform[:3, :3] = rotation

    translation_transform = np.eye(4)
    translation_transform[:3, 3] = -bf_mesh.centroid

    scale_transform = np.eye(4)
    scale_transform[0, 0] = -1.

    bf_mesh = bf_mesh.apply_transform(rotation_transform @ translation_transform @ scale_transform)

    bf_mesh.export('before-bf_mesh.ply')
    pcd.export('before-pcd.ply')

    sample_points = 2 ** 15
    bf_sampled_points, _ = trimesh.sample.sample_surface(bf_mesh, count=sample_points)

    transform, _, cost = trimesh.registration.icp(bf_sampled_points, points3d[sample_points::len(points3d) // sample_points], max_iterations=100, threshold=1e-10, scale=False, reflection=False)

    bf_mesh = bf_mesh.apply_transform(transform)

    bf_mesh.export('after-bf_mesh.ply')
    pcd.export('after-pcd.ply')
