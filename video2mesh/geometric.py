"""
This module contains functions and classes used for manipulating camera trajectories, projecting points between
2D image and 3D world coordinates, and creating point clouds.
"""
import dataclasses
from typing import Optional, Tuple, Dict

import numpy as np
import torch
from matplotlib import pyplot as plt
from scipy.interpolate import interp1d
from scipy.spatial.transform import Rotation, Slerp

from video2mesh.types import File, Size
from video2mesh.utils import validate_shape, validate_camera_parameter_shapes


def pose_vec2mat(pose: np.ndarray) -> np.ndarray:
    """
    Convert a transformation 7-vector [r, t] to a (4, 4) homogenous transformation matrix.

    :param pose: The 7-vector to convert.
    :return: The (4, 4) homogeneous transformation matrix.
    """
    validate_shape(pose, 'pose', expected_shape=(7,))
    R = Rotation.from_quat(pose[:4]).as_matrix()
    t = pose[4:].reshape((-1, 1))

    M = np.eye(4, dtype=R.dtype)
    M[:3, :3] = R
    M[:3, 3:] = t

    return M


def pose_mat2vec(pose: np.ndarray) -> np.ndarray:
    """
    Convert a homogenous transformation matrix to a transformation 7-vector [r, t].

    :param pose: The (4, 4) homogenous transformation matrix to convert.
    :return: The 7-vector transformation.
    """
    quaternion = Rotation.from_matrix(pose[:3, :3]).as_quat()
    translation_vector = pose[:3, 3]

    return np.hstack((quaternion, translation_vector))


def get_pose_components(pose):
    """
    Get the [R | t] components of a camera pose.

    :param pose: The (4, 4) homogenous camera intrinsics matrix.
    :return: A 2-tuple containing the (3, 3) rotation matrix R, and the (3, 1) translation vector.
    """
    validate_shape(pose, 'pose', (4, 4))

    R = pose[:3, :3]
    t = pose[:3, 3:]

    return R, t


def add_pose(pose_a, pose_b) -> np.ndarray:
    """
    Accumulate two poses.
    :param pose_a: The first (1, 7) pose.
    :param pose_b: The second (1, 7) pose.
    :return: The (1, 7) pose a + b.
    """
    return pose_mat2vec(pose_vec2mat(pose_b) @ pose_vec2mat(pose_a))


def subtract_pose(pose_a, pose_b) -> np.ndarray:
    """
    Get relative pose between two poses (i.e. `pose_a - pose_b`).

    :param pose_a: The first (1, 7) pose.
    :param pose_b: The second (1, 7) pose.
    :return: The (1, 7) relative pose between `pose_a` and `pose_b`.
    """
    return pose_mat2vec(np.linalg.inv(pose_vec2mat(pose_b)) @ pose_vec2mat(pose_a))


def get_identity_pose():
    """Get the identity 7-vector pose (quaternion + translation vector)."""
    return np.asarray([0., 0., 0., 1., 0., 0., 0.])


def point_cloud_from_depth(depth, mask, K, R=np.eye(3), t=np.zeros((3, 1))):
    """
    Create a point cloud from a depth map.

    :param depth: A depth map.
    :param mask: A binary mask the same shape as the depth maps.
        Truthy values indicate parts of the depth maps that should be kept.
    :param K: The camera intrinsics matrix.
    :param R: The (3, 3) rotation matrix.
    :param t: The (3, 1) translation vector.

    :return: the (N, 3) point cloud.
    """
    valid_pixels = mask & (depth > 0.0)
    V, U = valid_pixels.nonzero()
    points2d = np.array([U, V]).T

    points = image2world(points2d, depth[valid_pixels], K, R, t)

    return points


def point_cloud_from_rgbd(rgb, depth, mask, K, R=np.eye(3), t=np.zeros((3, 1))):
    """
    Create a point cloud with vertex colours from an RGB-D frame.

    :param rgb: A colour image.
    :param depth: A depth map.
    :param mask: A binary mask the same shape as the depth maps.
        Truthy values indicate parts of the depth maps that should be kept.
    :param K: The camera intrinsics matrix.
    :param R: The (3, 3) rotation matrix.
    :param t: The (3, 1) translation vector.

    :return: the (N, 3) point cloud.
    """
    valid_pixels = mask & (depth > 0.0)
    V, U = valid_pixels.nonzero()
    points2d = np.array([U, V]).T

    points = image2world(points2d, depth[valid_pixels], K, R, t)
    colour = np.zeros(shape=(len(points), 4), dtype=rgb.dtype)
    colour[:, :3] = rgb[valid_pixels]
    colour[:, 3] = 255

    return points, colour


def world2image(points, K, R=np.eye(3), t=np.zeros((3, 1)), scale_factor=1.0, dtype=np.int32):
    """
    Convert 3D world coordinates to 2D image coordinates.

    :param points: The (?, 3) array of world coordinates.
    :param K: The (3, 3) camera intrinsics matrix.
    :param R: The (3, 3) camera rotation matrix.
    :param t: The (3, 1) camera translation column vector.
    :param scale_factor: An optional value that scales the 2D points.
    :param dtype: The data type of the returned points.

    :return: a 2-tuple containing: the (?, 2) 2D points in image space; the recovered depth of the 2D points.
    """
    validate_shape(points, 'points', expected_shape=(None, 3))
    validate_camera_parameter_shapes(K, R, t)

    camera_space_coords = K @ (R @ points.T + t)
    depth = camera_space_coords[2, :]
    pixel_coords = camera_space_coords[0:2, :] / depth / scale_factor

    if issubclass(dtype, np.integer):
        pixel_coords = np.round(pixel_coords)

    pixel_coords = np.array(pixel_coords.T, dtype=dtype)

    return pixel_coords, depth


def image2world(points, depth, K, R=np.eye(3), t=np.zeros((3, 1)), scale_factor=1.0):
    """
    Convert 2D image coordinates to 3D world coordinates.

    :param points: The (N, 2) array of image coordinates.
    :param depth: The (N,) array of depth values at the given 2D points.
    :param K: The (3, 3) camera intrinsics matrix.
    :param R: The (3, 3) camera rotation matrix.
    :param t: The (3, 1) camera translation column vector.
    :param scale_factor: An optional value that scales the 2D points.

    :return: the (N, 3) 3D points in world space.
    """
    validate_shape(points, 'points', expected_shape=(None, 2))
    validate_shape(depth, 'depth', expected_shape=(points.shape[0],))
    validate_camera_parameter_shapes(K, R, t)

    num_points = points.shape[0]

    points2d = np.vstack((points.T * scale_factor, np.ones(num_points)))
    points_camera_space = np.linalg.inv(K) @ points2d
    points_world_space = R.T @ (depth * points_camera_space - t)

    return points_world_space.T


class Quaternion:
    """Implements basic quaterion-quaternion and quaternion-vector operations."""

    def __init__(self, values: torch.Tensor):
        """
        :param values: The 4xN matrix of quaternions where the rows are the x, y, z, and w components.
        """
        if len(values.shape) != 2 or values.shape[0] != 4:
            raise ValueError(f"Invalid shape. Expected shape (4, N) but got {values.shape}.")

        self.values = values

    @property
    def x(self):
        return self.values[0]

    @property
    def y(self):
        return self.values[1]

    @property
    def z(self):
        return self.values[2]

    @property
    def w(self):
        return self.values[3]

    def __mul__(self, other):
        if isinstance(other, Quaternion):
            return Quaternion.multiply(self, other)
        else:
            raise TypeError(f"Cannot multiply a {self.__class__.__name__} with a {type(other)}")

    def __rmul__(self, other):
        return self.__mul__(other)

    def conjugate(self) -> 'Quaternion':
        """Get the conjugate of a quaternion (-x, -y, -z, w)."""
        return Quaternion(torch.vstack((-self.x, -self.y, -self.z, self.w)))

    def inverse(self) -> 'Quaternion':
        """
        Get the inverse rotation (i.e. the conjugate).
        Alias for `.conjugate()`.
        :return: The inverse rotation quaternion.
        """
        return self.conjugate()

    def normalise(self) -> 'Quaternion':
        """
        Normalise a quaternion.
        :return: A unit quaternion.
        """
        norm = torch.linalg.norm(self.values, ord=2, dim=0)

        return Quaternion(self.values / norm)

    @staticmethod
    def multiply(q1: 'Quaternion', q2: 'Quaternion') -> 'Quaternion':
        """
        Multiply two quaternions together.

        :param q1: The first quaternion to multiply.
        :param q2: The second quaternion to multiply.
        :return: The result of multiplying the two quaternions.
        """
        x1, y1, z1, w1 = q1.values
        x2, y2, z2, w2 = q2.values
        x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
        y = w1 * y2 + y1 * w2 + z1 * x2 - x1 * z2
        z = w1 * z2 + z1 * w2 + x1 * y2 - y1 * x2
        w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2

        return Quaternion(torch.vstack((x, y, z, w)))

    def apply(self, v: torch.Tensor) -> torch.Tensor:
        """
        Apply the rotation to a vector.

        :param v: The vector to rotate in (3, N) format.
        :return: The rotated vector in (3, N) format.
        """
        assert len(v.shape) == 2 and v.shape[0] == 3

        q = Quaternion(torch.vstack((v, torch.zeros(v.shape[1], dtype=v.dtype, device=v.device))))

        return (self * q * self.conjugate()).values[:3, :]

    def __repr__(self):
        return f"{self.__class__.__name__}({repr(self.values)})"


class Trajectory:
    """Encapsulates a sequence of camera poses."""

    def __init__(self, values: Optional[np.ndarray] = None):
        """
        :param values: The Nx7 camera poses where each pose (row) is a 7-vector consisting of:
            *  a scalar-last quaternion
            * and camera's XYZ position.
        """
        if values is not None:
            validate_shape(values, 'values', (None, 7))

        self.values = values

    def __len__(self):
        return len(self.values)

    def __getitem__(self, index):
        return self.values[index]

    def __setitem__(self, index, value):
        self.values[index] = value

    def __iter__(self):
        return iter(self.values)

    @property
    def rotations(self) -> np.ndarray:
        return self.values[:, :4]

    @property
    def positions(self) -> np.ndarray:
        return self.values[:, 4:]

    @property
    def shape(self) -> tuple:
        return self.values.shape

    def copy(self) -> 'Trajectory':
        """
        :return: Get a copy of the trajectory.
        """
        return Trajectory(self.values.copy())

    def save(self, f: File):
        """
        Save the trajectory to disk.

        :param f: The file (path string or file object) to save to.
        """
        # noinspection PyTypeChecker
        np.savetxt(f, self.values)

    @classmethod
    def load(cls, f: File) -> 'Trajectory':
        """
        Load a trajectory from disk.

        :param f: The file (path string or file object) to load.
        :return: The loaded trajectory.
        """
        values = np.loadtxt(f, dtype=np.float32)

        if len(values.shape) == 1:
            # Convert from flat array to 2D array.
            values = values.reshape((1, -1))

        return Trajectory(values)

    def normalise(self) -> 'Trajectory':
        """
        Adjust the camera trajectory so that the first pose is the identity.

        :return: The normalised camera trajectory.
        """
        matrix_trajectory = self.to_homogenous_transforms()
        matrix_trajectory = np.linalg.inv(matrix_trajectory[0]) @ matrix_trajectory
        vector_trajectory = self.from_homogenous_transforms(matrix_trajectory)

        return vector_trajectory

    def normalise_position(self) -> 'Trajectory':
        """
        Adjust the camera trajectory so that the translation is the origin.

        :return: The normalised camera trajectory.
        """
        matrix_trajectory = self.to_homogenous_transforms()

        first_pose = matrix_trajectory[0].copy()
        first_pose[:3, :3] = np.eye(3)
        matrix_trajectory = np.linalg.inv(first_pose) @ matrix_trajectory

        vector_trajectory = self.from_homogenous_transforms(matrix_trajectory)

        return vector_trajectory

    def inverse(self) -> 'Trajectory':
        """
        Get the inverse of the camera trajectory.

        :return: The inverted camera trajectory.
        """
        matrix_trajectory = self.to_homogenous_transforms()
        matrix_trajectory = np.linalg.inv(matrix_trajectory)
        vector_trajectory = self.from_homogenous_transforms(matrix_trajectory)

        return vector_trajectory

    def apply(self, transform: np.ndarray) -> 'Trajectory':
        """
        Apply a transformation to the camera trajectory (i.e. to each pose).

        :param transform: A 4x4 homogeneous transformation matrix.
        :return: The transformed trajectory.
        """
        matrix_trajectory = self.to_homogenous_transforms()
        matrix_trajectory_transformed = matrix_trajectory @ transform
        vector_trajectory = self.from_homogenous_transforms(matrix_trajectory_transformed)

        return vector_trajectory

    def tensor(self) -> torch.FloatTensor:
        """
        Get the trajectory as a Pytorch Tensor.

        :return: A Nx7 float32 tensor.
        """
        return torch.from_numpy(self.values).to(torch.float32)

    def calculate_ate(self, other: 'Trajectory') -> np.ndarray:
        """
        Calculate ATE (Absolute Trajectory Error) between this trajectory and another.

        :param other: The other trajectory.
        :return: The alignment error per frame.
        """
        if len(self) != len(other):
            raise RuntimeError(f"Got trajectories of unequal length ({len(self)} and {len(other)})")

        # Adapted from: https://github.com/tinghuiz/SfMLearner/blob/master/kitti_eval/pose_evaluation_utils.py
        trajectory_normalised = self.normalise().positions
        other_normalised = other.normalise().positions

        scale = np.sum(trajectory_normalised * other_normalised) / np.sum(np.square(other_normalised))
        alignment_error = other_normalised * scale - trajectory_normalised

        return alignment_error

    def calculate_rpe(self, other: 'Trajectory') -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate the RPE (Relative Pose Error) between this trajectory and another.

        :param other: The other trajectory.
        :return: A 2-tuple containing the rotational error (radians) and translational error (meters) per frame,
            respectively.
        """
        if len(self) != len(other):
            raise RuntimeError(f"Got trajectories of unequal length ({len(self)} and {len(other)})")

        rotational_error = []
        translational_error = []

        num_frames = len(self)

        gt = self.normalise().to_homogenous_transforms()
        pred = other.normalise().to_homogenous_transforms()

        for i, j in zip(range(num_frames - 1), range(1, num_frames)):
            rel_pose_est = np.linalg.inv(pred[i]) @ pred[j]
            rel_pose_gt = np.linalg.inv(gt[i]) @ gt[j]
            rel_pose_error = np.linalg.inv(rel_pose_gt) @ rel_pose_est

            distance = np.linalg.norm(rel_pose_error[:3, 3])
            # The below is equivalent to converting to axis-angle or Euler angles and taking the L2 norm.
            angle = np.arccos(min(1, max(-1, (np.trace(rel_pose_error[:3, :3]) - 1) / 2)))

            translational_error.append(distance)
            rotational_error.append(angle)

        rotational_error = np.asarray(rotational_error)
        translational_error = np.asarray(translational_error)

        return rotational_error, translational_error

    def plot(self, output_path: Optional[str] = None):
        """
        Plot a camera trajectory (camera positions).

        :param output_path: (optional) The file to save the plot to. If `None`, the plot will be displayed on screen.
        """
        trajectory = self.normalise().positions
        _, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.8, 4.8))

        self._plot_trajectory(trajectory, plot_axis=ax1, secondary_axis='y')
        self._plot_trajectory(trajectory, plot_axis=ax2, secondary_axis='z')

        plt.tight_layout()

        if output_path is None:
            plt.show()
        else:
            plt.savefig(output_path, dpi=90)

        plt.close()

    def plot_comparison(self, other: 'Trajectory', output_path: Optional[str] = None):
        """
        Plot two trajectories (camera positions) over each other for comparison.

        :param other: The other trajectory.
        :param output_path: (optional) The file to save the plot to. If `None`, the plot will be displayed on screen.
        """
        if len(self) != len(other):
            raise RuntimeError(f"Got trajectories of unequal length ({len(self)} and {len(other)})")

        _, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.8, 4.8))

        gt_trajectory = self.normalise().positions
        pred_trajectory = other.normalise().positions

        self._plot_trajectory(gt_trajectory, pred_trajectory, plot_axis=ax1, secondary_axis='y')
        self._plot_trajectory(gt_trajectory, pred_trajectory, plot_axis=ax2, secondary_axis='z')

        plt.tight_layout()

        if output_path is None:
            plt.show()
        else:
            plt.savefig(output_path, dpi=90)

        plt.close()

    @staticmethod
    def _plot_trajectory(gt_trajectory: np.ndarray, pred_trajectory: Optional[np.ndarray] = None,
                         plot_axis: Optional[plt.Axes] = None, secondary_axis='y'):
        """
        Plot a trajectory (camera positions) on a 2D plane.

        :param gt_trajectory: The reference trajectory.
        :param pred_trajectory: (optional) A trajectory to plot alongside the reference trajectory for comparison.
        :param plot_axis: (optional) The plot axis (matplotlib.pyplot.Axes object) to plot to.
        :param secondary_axis: The secondary axis which defines which plane to plot the trajectory on.
            The primary axis is always the x-axis. The XZ plane is the ground plane and the XZ plane is a vertical plane.
        """
        if plot_axis is None:
            plot_axis = plt.gca()

        if secondary_axis == 'y':
            axis = 1
        elif secondary_axis == 'z':
            axis = 2
        else:
            raise RuntimeError(f"secondary_axis must be one of ('y', 'z').")

        if pred_trajectory is None:
            plot_axis.plot(gt_trajectory[:, 0], gt_trajectory[:, axis], '-', color="black")
        else:
            plot_axis.plot(gt_trajectory[:, 0], gt_trajectory[:, axis], '-', color="black", label="ground truth")
            plot_axis.plot(pred_trajectory[:, 0], pred_trajectory[:, axis], '-', color="blue", label="estimated")
            plot_axis.legend()

        plot_axis.set_xlabel("x [m]")
        plot_axis.set_ylabel(f"{secondary_axis} [m]")
        plot_axis.set_title(f"Trajectory on X{secondary_axis.upper()} Plane")

    def to_homogenous_transforms(self) -> np.ndarray:
        """
        Convert each pose in a camera trajectory from a quaternion + translation vector to a homogenous 4x4 transformation.

        :return: The (N, 4, 4) camera trajectory.
        """
        T = np.tile(np.eye(4), (len(self), 1, 1))
        T[:, :3, :3] = Rotation.from_quat(self.rotations).as_matrix()
        T[:, :3, 3] = self.positions

        return T

    @staticmethod
    def from_homogenous_transforms(camera_trajectory: np.ndarray) -> 'Trajectory':
        """
        Convert each pose in a camera trajectory from a homogenous 4x4 transformation to a quaternion + translation vector.

        :param camera_trajectory: The (N, 4, 4) camera trajectory to adjust.
            Each row should be a 4x4 homogenous transformation matrix.
        :return: The (N, 7) camera trajectory where each row is a quaternion and translation vector.
        """
        validate_shape(camera_trajectory, 'camera_trajectory', (None, 4, 4))

        r = Rotation.from_matrix(camera_trajectory[:, :3, :3]).as_quat()
        t = camera_trajectory[:, :3, 3]
        values = np.hstack((r, t))

        return Trajectory(values)

    @staticmethod
    def create_by_interpolating(poses: Dict[int, np.ndarray], frame_count: int) -> 'Trajectory':
        """
        Create a new `Trajectory` object by interpolating some pose data.

        Interpolate the pose for frames with missing data (when `frame_step` > 1).

        :param poses: A mapping between the input frame index and a pose vector.
        :param frame_count: The total frames in the sequence.
        :return: The interpolated Nx7 camera poses.
        """
        if 0 not in poses:
            raise RuntimeError("Cannot interpolate trajectory where the pose for the first frame is missing.")

        if frame_count - 1 not in poses:
            raise RuntimeError("Cannot interpolate trajectory where the pose for the last frame is missing.")

        frames_with_pose = sorted(poses.keys())
        start_and_end_indices = zip(frames_with_pose[:-1], frames_with_pose[1:])

        interpolated_poses = np.zeros((frame_count, 7))

        for (start_index, end_index) in start_and_end_indices:
            start_rotation = poses[start_index][:4]
            end_rotation = poses[end_index][:4]

            start_position = poses[start_index][4:]
            end_position = poses[end_index][4:]

            key_frame_times = [0, 1]
            num_frames_to_interpolate = (end_index + 1) - start_index
            times_to_interpolate = np.linspace(0, 1, num=num_frames_to_interpolate)

            slerp = Slerp(times=key_frame_times, rotations=Rotation.from_quat([start_rotation, end_rotation]))
            lerp = interp1d(key_frame_times, [start_position, end_position], axis=0)

            interpolated_poses[start_index:end_index + 1, 4:] = lerp(times_to_interpolate)
            interpolated_poses[start_index:end_index + 1, :4] = slerp(times_to_interpolate).as_quat()

        return Trajectory(interpolated_poses)


@dataclasses.dataclass(frozen=True)
class CameraMatrix:
    """A 3x3 camera matrix that models a simple pinhole camera"""

    """Focal length x"""
    fx: float
    """Focal length y"""
    fy: float
    """Optical center x"""
    cx: float
    """Optical center y"""
    cy: float
    """Sensor width (pixels)"""
    width: int
    """Sensor height (pixels)"""
    height: int

    @property
    def matrix(self) -> np.ndarray:
        """Get the 3x3 camera matrix as a NumPy array."""
        return np.array([
            [self.fx, 0., self.cx],
            [0., self.fy, self.cy],
            [0., 0., 1.]
        ])

    def scale(self, target_size: Size) -> 'CameraMatrix':
        """
        Get the camera matrix for the Kinect Sensor.

        :param target_size: The (height, width) to rescale the camera parameters to.
        :return: The scaled camera matrix.
        """
        target_height, target_width = target_size
        scale_x = target_width / self.width
        scale_y = target_height / self.height

        return CameraMatrix(
            fx=self.fx * scale_x,
            fy=self.fy * scale_y,
            cx=self.cx * scale_x,
            cy=self.cy * scale_y,
            width=target_width,
            height=target_height
        )
