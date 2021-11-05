import argparse
import os

import cv2
import numpy as np

from Video2mesh.geometry import pose_vec2mat, get_pose_components, point_cloud_from_depth
from Video2mesh.io import load_camera_parameters, load_input_data, numpy_to_ply, write_ply
from Video2mesh.options import StorageOptions, DepthOptions, DepthFormat
from Video2mesh.utils import Timer


def _extract_match_data(frame_i, frame_j, depth_i, depth_j, mask_i=None, mask_j=None,
                        ratio_threshold=0.7, ransac_threshold=4):
    """
    Extract matching feature points and the corresponding depth values in a pair of RGB-D frames.

    :param frame_i: The first RGB frame.
    :param frame_j: The other RGB frame.
    :param depth_i: The first depth map.
    :param depth_j: The other depth map.
    :param mask_i: A binary mask for the first frame where False/0 pixels indicate areas that should be ignored.
    :param mask_j: A binary mask for the other frame where False/0 pixels indicate areas that should be ignored.
    :param ratio_threshold: The ratio to use in Lowe's ratio test when keeping/rejecting possible feature matches.
    :param ransac_threshold: The minimum number of matched features needed to refine matched features with RANSAC.

    :return: A 4-tuple containing: a `MatchData` object for the feature points and corresponding depth values for
        each frame; the number of matches discarded due to missing depth values; and a visualisation of the image
        features and matches between the two frames.
    """
    sift = cv2.SIFT_create()
    flann = cv2.FlannBasedMatcher_create()

    frame1 = cv2.cvtColor(frame_i, cv2.COLOR_RGB2GRAY)
    frame2 = cv2.cvtColor(frame_j, cv2.COLOR_RGB2GRAY)

    # find the keypoints and descriptors with SIFT
    key_points_1, des1 = sift.detectAndCompute(frame1, mask_i)
    key_points_2, des2 = sift.detectAndCompute(frame2, mask_j)

    matches = flann.knnMatch(des1, des2, k=2)
    # Need to draw only good matches, so create a mask
    matchesMask = np.array([[0, 0] for _ in range(len(matches))])

    num_missing_depth = 0

    a_points = []
    b_points = []
    a_depth = []
    b_depth = []

    for k, (m, n) in enumerate(matches):
        # ratio test as per Lowe's paper
        if m.distance / n.distance < ratio_threshold:
            p = np.array(key_points_1[m.queryIdx].pt)
            u_p, v_q = np.round(p).astype(int)
            depth_p = depth_i[v_q, u_p]

            q = np.array(key_points_2[m.trainIdx].pt)
            u_q, v_q = np.round(q).astype(int)
            depth_q = depth_j[v_q, u_q]

            if depth_p == 0.0 or depth_q == 0.0:
                num_missing_depth += 1

                continue

            matchesMask[k] = [1, 0]

            a_points.append(p)
            a_depth.append(depth_p)

            b_points.append(q)
            b_depth.append(depth_q)

    num_good_matches = len(a_points)

    print(f"\tFound {num_good_matches} good matches.")
    print(f"\t{num_missing_depth} matches were discarded due to missing depth.")

    if num_good_matches > ransac_threshold:
        print(f"\tRefining matches with RANSAC...")
        a_points = np.array(a_points)
        b_points = np.array(b_points)
        a_depth = np.array(a_depth)
        b_depth = np.array(b_depth)

        _, mask = cv2.findHomography(a_points, b_points, cv2.RANSAC)

        num_inliers = mask.sum()
        print(f"\tFound {num_inliers:,d} inliers out of {len(mask):,d} matched points.")

        is_inlier = mask.flatten() > 0
        # Need to undo the matchesMask for good matches that were not inliers to ensure the viz is correct.
        matchesMask[np.argwhere((matchesMask == [1, 0]).all(axis=1))[~is_inlier]] = [0, 0]

        a_points = a_points[is_inlier].tolist()
        a_depth = a_depth[is_inlier].tolist()

        b_points = b_points[is_inlier].tolist()
        b_depth = b_depth[is_inlier].tolist()

    draw_params = dict(matchColor=(0, 255, 0),
                       singlePointColor=(255, 0, 0),
                       matchesMask=matchesMask,
                       flags=cv2.DrawMatchesFlags_DEFAULT)

    kp_matches_viz = cv2.drawMatchesKnn(frame_i, key_points_1, frame_j, key_points_2, matches, None, **draw_params)
    kp_matches_viz = cv2.cvtColor(kp_matches_viz, cv2.COLOR_BGR2RGB)

    a_points = np.asarray(a_points)
    a_depth = np.asarray(a_depth)
    b_points = np.asarray(b_points)
    b_depth = np.asarray(b_depth)

    return (a_points, a_depth), (b_points, b_depth), kp_matches_viz


def convert_to_depth_to_plane(depth_map, f):
    _, h, w = depth_map.shape
    i_c = float(h) / 2 - 1
    j_c = float(w) / 2 - 1
    columns, rows = np.meshgrid(np.linspace(0, w - 1, num=w), np.linspace(0, h - 1, num=h))
    distance_from_center = ((rows - i_c) ** 2 + (columns - j_c) ** 2) ** 0.5
    depth_map_plane = depth_map / (1 + (distance_from_center / f) ** 2) ** 0.5

    return depth_map_plane


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Reconstruct the static background of a RGB-D video sequence.")

    StorageOptions.add_args(parser)
    DepthOptions.add_args(parser)

    args = parser.parse_args()
    print(args)

    storage_options = StorageOptions.from_args(args)
    depth_options = DepthOptions.from_args(args)

    K, trajectory = load_camera_parameters(storage_options)
    print(K.shape, trajectory.shape)

    # Convert from Unreal Engine coordinate system
    trajectory = trajectory[:, [0, 1, 2, 3, 5, 4]]

    rgb_frames, depth_maps, masks = load_input_data(storage_options, depth_options)
    print(rgb_frames.shape, depth_maps.shape, masks.shape)

    if depth_options.depth_format == DepthFormat.DEPTH_TO_POINT:
        assert K[0, 0] == K[1, 1]
        depth_maps = convert_to_depth_to_plane(depth_maps, K[0, 0])

    i = 1
    j = i + 1

    (points_a, depth_a), (points_b, depth_b), _ = _extract_match_data(rgb_frames[i], rgb_frames[j], depth_maps[i],
                                                                      depth_maps[j])


    # pose_i = pose_vec2mat(trajectory[i])
    pose_i = np.eye(4)
    R_1, t_1 = get_pose_components(pose_i)

    pose_j = pose_vec2mat(trajectory[j])
    pose_j = np.linalg.inv(pose_vec2mat(trajectory[i])) @ pose_j
    R_2, t_2 = get_pose_components(pose_j)

    K_inv = np.linalg.inv(K)


    def point2d_to_homogenous(x):
        return np.vstack((np.transpose(x), np.ones(len(x))))

    def camera_to_image_plane(x):
        return x[:2] / x[2]

    points_a_world = R_1.T @ (depth_a * (K_inv @ point2d_to_homogenous(points_a)) - t_1)
    points_b_world = R_2.T @ (depth_b * (K_inv @ point2d_to_homogenous(points_b)) - t_2)
    points_a_in_b = camera_to_image_plane(K @ (R_2 @ points_a_world + t_2)).T

    # TODO: See if matched points can be projected into the same places with GT pose
    def get_pcd(k):
        frame = rgb_frames[k]
        depth_map = depth_maps[k]

        mask = masks[k] == 0
        pose = pose_vec2mat(trajectory[k])
        pose_origin = pose_vec2mat(trajectory[0])
        pose = np.linalg.inv(pose_origin) @ pose
        R, t = get_pose_components(pose)
        I, J = (mask & (depth_map > 0)).nonzero()
        point_cloud = point_cloud_from_depth(depth_map, mask, K, R, t)
        color3d = frame[I, J, :]

        print(point_cloud.shape, point_cloud.min(axis=0), point_cloud.max(axis=0), point_cloud.mean(axis=0))
        print(color3d.shape)

        return point_cloud, color3d


    pcd1, colour1 = get_pcd(i)
    pcd2, colour2 = get_pcd(j)

    point_cloud = np.concatenate((pcd1, pcd2))
    colour3d = np.concatenate((colour1, colour2))

    t = Timer()

    ply_data = numpy_to_ply(point_cloud, colour3d)
    t.split("create ply data")

    write_ply(os.path.join(storage_options.base_folder, "pcd.ply"), ply_data)
    t.split("write ply data")
