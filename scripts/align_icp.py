import os.path

import argparse
import datetime
import numpy as np
import trimesh
from scipy.spatial.transform import Rotation
from tqdm import tqdm

from Video2mesh.geometry import pose_vec2mat, point_cloud_from_depth, point_cloud_from_rgbd
from Video2mesh.io import VTMDataset
from Video2mesh.utils import log


def get_dataset(args):
    dataset_path = args.dataset_path

    VTMDataset._validate_dataset(dataset_path)

    dataset = VTMDataset(dataset_path) \
        .create_or_find_masks()

    return dataset


def get_scene_bounds_and_centroid(dataset, num_frames):
    bounds = np.zeros((2, 3), dtype=float)
    i = 0

    for depth_map, pose, mask_encoded in \
            tqdm(zip(dataset.depth_dataset, dataset.camera_trajectory, dataset.mask_dataset), total=num_frames):
        if i >= num_frames:
            break
        else:
            i += 1

        binary_mask = mask_encoded == 0

        pose_matrix = pose_vec2mat(pose)
        pose_matrix = np.linalg.inv(pose_matrix)
        R = pose_matrix[:3, :3]
        t = pose_matrix[:3, 3:]

        points3d = point_cloud_from_depth(depth_map, binary_mask, dataset.camera_matrix, R, t)

        # Expand bounds
        bounds[0] = np.min(np.vstack((bounds[0], points3d.min(axis=0))), axis=0)
        bounds[1] = np.max(np.vstack((bounds[1], points3d.max(axis=0))), axis=0)

    centroid = bounds.mean(axis=0)

    return bounds, centroid

def print_bounds_comparison_stats(scene_bounds, bf_bounds):
    scene_centroid = scene_bounds.mean(axis=0)
    bf_centroid = bf_bounds.mean(axis=0)

    centroid_distance = np.sqrt(np.mean(np.square(scene_centroid - bf_centroid)))
    log(f"Centroid distance: {centroid_distance:.2f} m")
    # Compare coverage (range of bounds in scene bounds vs mesh bounds)
    scene_range = scene_bounds[1] - scene_bounds[0]
    bf_range = bf_bounds[1] - bf_bounds[0]

    percentage_difference = 100 * (bf_range - scene_range) / scene_range
    log(f"Percent difference of the range covered by the ground truth scene and "
        f"the BundleFusion reconstruction across each of the XYZ axes:")
    log(', '.join(list(map(lambda x: f"{x:.1f}%", percentage_difference))))

    log(f"Area of:")
    scene_area = np.product(scene_range)
    bf_area = np.product(bf_range)
    area_percent_difference = 100 * (bf_area - scene_area) / scene_area
    log(f"\tGround Truth Scene: {scene_area :.2f} m^3")
    log(f"\tBundleFusion Scene: {bf_area :.2f} m^3")
    log(f"Percent difference of scene areas: {area_percent_difference:.1f}%")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', type=str, help='Path to the RGB-D dataset.')
    parser.add_argument('--num_frames', type=int, default=-1)

    args = parser.parse_args()

    start = datetime.datetime.now()

    log("Load dataset")
    dataset = get_dataset(args)

    num_frames = args.num_frames if args.num_frames > 0 else dataset.num_frames

    log("Calculating scene bounds...")
    bounds, centroid = get_scene_bounds_and_centroid(dataset, num_frames)

    # Load BF mesh.
    bf_mesh_path = os.path.join(dataset.base_path, 'mesh', 'bg.glb')
    bf_mesh = trimesh.load(bf_mesh_path)

    # Move mesh s.t. its centroid equals the scene centroid
    transform = np.eye(4)
    transform[:3, 3] = bf_mesh.centroid - centroid
    transform[0, 0] = -1  # BundleFusion reconstruction is flipped horizontally, this flips it back.
    bf_mesh.apply_transform(transform)
    bf_mesh.apply_translation([0., 0., 1.])
    bf_mesh.geometry['geometry_0'].export('bf_mesh-before.ply')
    print_bounds_comparison_stats(bounds, bf_mesh.bounds)

    # Find transform to align mesh to first frame's point cloud
    log(f"Running ICP...")
    index = 0
    frame = dataset.rgb_dataset[index]
    depth_map = dataset.depth_dataset[index]
    pose = dataset.camera_trajectory[index]
    mask_encoded = dataset.mask_dataset[index]
    binary_mask = mask_encoded == 0

    rgb = np.ascontiguousarray(frame[:, :, :3])
    pose_matrix = pose_vec2mat(pose)
    pose_matrix = np.linalg.inv(pose_matrix)
    R = pose_matrix[:3, :3]
    t = pose_matrix[:3, 3:]

    sample_points = 2 ** 15
    max_depth = 5.0
    depth_map[depth_map > max_depth] = 0.
    colour_points, points3d = point_cloud_from_rgbd(rgb, depth_map, binary_mask, dataset.camera_matrix, R, t)
    pcd = trimesh.PointCloud(points3d, colour_points)

    sampling_interval = max(1, round(len(points3d) / sample_points))
    sampled_points = points3d[::sampling_interval]
    bf_sampled_points, _ = trimesh.sample.sample_surface(bf_mesh.geometry['geometry_0'], count=len(sampled_points))

    transform, _, cost = trimesh.registration.icp(bf_sampled_points, sampled_points,
                                                  max_iterations=100, threshold=1e-15, scale=False, reflection=False)

    bf_pcd = trimesh.PointCloud(bf_sampled_points)
    bf_pcd.export('bf_pcd-before.ply')
    bf_pcd.apply_transform(transform)
    bf_pcd.export('bf_pcd.ply')
    bf_mesh.apply_transform(transform)

    log("Stats AFTER ICP:")
    print_bounds_comparison_stats(bounds, bf_mesh.bounds)

    # Apply 180 rotation so that mesh is facing the correct direction in MeshLab
    rot_180 = Rotation.from_euler('xyz', [0, 180, 0], degrees=True).as_matrix()
    the_rotation = np.eye(4, dtype=np.float32)
    the_rotation[:3, :3] = rot_180

    bf_mesh.apply_transform(the_rotation)
    pcd.apply_transform(the_rotation)

    # Export mesh
    bf_mesh.geometry['geometry_0'].export('bf_mesh.ply')
    pcd.export('pcd.ply')
