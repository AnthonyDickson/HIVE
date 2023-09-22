"""This module defines the camera parameters for common cameras/sensors."""
import numpy as np


class KinectSensor:
    """The Kinect RGB-D sensor used in the NYU v2 RGB-D dataset and the TUM RGB-D dataset."""
    fx = 580.0  # focal length x
    fy = 580.0  # focal length y
    cx = 319.5  # optical center x
    cy = 239.5  # optical center y
    width = 640
    height = 480

    @classmethod
    def get_camera_matrix(cls) -> np.ndarray:
        return np.array([[cls.fx, 0., cls.cx],
                         [0., cls.fy, cls.cy],
                         [0., 0., 1.]])