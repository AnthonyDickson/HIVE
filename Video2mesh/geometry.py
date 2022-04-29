import cv2
import numpy as np
import torch
from scipy.spatial.transform import Rotation

from Video2mesh.options import MaskDilationOptions
from Video2mesh.utils import validate_shape, validate_camera_parameter_shapes


def pose_vec2mat(pose):
    """
    Convert a transformation 7-vector [r, t] to a (4, 4) homogenous transformation matrix.

    :param pose: The 7-vector to convert.
    :return: The (4, 4) homogeneous transformation matrix.
    """
    validate_shape(pose, 'pose', expected_shape=(7,))
    R = Rotation.from_quat(pose[:4]).as_matrix()
    t = pose[4:].reshape((-1, 1))

    M = np.hstack((R, t))
    M = np.vstack((M, np.zeros(4)))
    M[-1, -1] = 1

    return M


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


def point_cloud_from_depth(depth, mask, K, R=np.eye(3), t=np.zeros((3, 1)), scale_factor=1.0):
    """

    :param depth: A depth map.
    :param mask: A binary mask the same shape as the depth maps.
        Truthy values indicate parts of the depth maps that should be kept.
    :param K: The camera intrinsics matrix.
    :param R: The (3, 3) rotation matrix.
    :param t: The (3, 1) translation vector.
    :param scale_factor:
    :return:
    """
    valid_pixels = mask & (depth > 0.0)
    V, U = valid_pixels.nonzero()
    points2d = np.array([U, V]).T

    points = image2world(points2d, depth[valid_pixels], K, R, t, scale_factor)

    return points


def point_cloud_from_rgbd(rgb, depth, mask, K, R=np.eye(3), t=np.zeros((3, 1)), scale_factor=1.0):
    valid_pixels = mask & (depth > 0.0)
    V, U = valid_pixels.nonzero()
    points2d = np.array([U, V]).T

    points = image2world(points2d, depth[valid_pixels], K, R, t, scale_factor)
    colour = np.zeros(shape=(len(points), 4), dtype=rgb.dtype)
    colour[:, :3] = rgb[valid_pixels]
    colour[:, 3] = 255

    return colour, points


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

    :param points: The (?, 2) array of image coordinates.
    :param depth: The (?,) array of depth values at the given 2D points.
    :param K: The (3, 3) camera intrinsics matrix.
    :param R: The (3, 3) camera rotation matrix.
    :param t: The (3, 1) camera translation column vector.
    :param scale_factor: An optional value that scales the 2D points.

    :return: the (?, 3) 3D points in world space.
    """
    validate_shape(points, 'points', expected_shape=(None, 2))
    validate_shape(depth, 'depth', expected_shape=(points.shape[0],))
    validate_camera_parameter_shapes(K, R, t)

    num_points = points.shape[0]

    points2d = np.vstack((points.T * scale_factor, np.ones(num_points)))
    pixel_i = np.linalg.inv(K) @ points2d
    pixel_world = R.T @ (depth * pixel_i - t)

    return pixel_world.T


def dilate_mask(mask, dilation_options: MaskDilationOptions):
    """
    Dilate an instance segmentation mask so that it covers a larger area.

    :param mask: The mask to enlarge/dilate.
    :param dilation_options: The object containing the dilation mask/filter and other settings.

    :return: The dilated mask.
    """
    validate_shape(mask, 'mask', expected_shape=(None, None))

    mask = mask.astype(np.float32)
    mask = cv2.dilate(mask.astype(float), dilation_options.filter, iterations=dilation_options.num_iterations)
    mask = mask.astype(bool)

    return mask


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

    def normalise(self, tolerance=0.00001) -> 'Quaternion':
        """
        Normalise a quaternion.
        :param tolerance:
        :return: A unit quaternion.
        """
        values = self.values
        norm = torch.linalg.norm(values, ord=2, dim=0)
        # noinspection PyTypeChecker
        values = torch.where(torch.abs(norm - 1.0) > tolerance, values / norm, values)

        return Quaternion(values)

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
