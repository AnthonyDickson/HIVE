import argparse
import os

from Video2mesh.geometry import pose_vec2mat, get_pose_components, point_cloud_from_depth
from Video2mesh.io import load_camera_parameters, load_input_data, numpy_to_ply, write_ply
from Video2mesh.options import StorageOptions
from Video2mesh.utils import Timer


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Reconstruct the static background of a RGB-D video sequence.")

    StorageOptions.add_args(parser)

    args = parser.parse_args()
    print(args)

    storage_options = StorageOptions.from_args(args)

    K, trajectory = load_camera_parameters(storage_options)
    print(K.shape, trajectory.shape)

    rgb_frames, depth_maps, masks = load_input_data(storage_options)
    print(rgb_frames.shape, depth_maps.shape, masks.shape)

    frame = rgb_frames[0]
    depth_map = depth_maps[0]
    mask = masks[0] == 0
    pose = pose_vec2mat(trajectory[0])
    R, t = get_pose_components(pose)

    I, J = (mask & (depth_map > 0)).nonzero()
    point_cloud = point_cloud_from_depth(depth_map, mask, K, R, t)
    color3d = frame[I, J, :]
    print(point_cloud.shape, point_cloud.min(axis=0), point_cloud.max(axis=0))
    print(color3d.shape)

    t = Timer()

    ply_data = numpy_to_ply(point_cloud, color3d)
    t.split("create ply data")

    write_ply(os.path.join(storage_options.base_folder, "pcd.ply"), ply_data)
    t.split("write ply data")
