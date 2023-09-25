"""This module defines the camera parameters for common cameras/sensors."""

from video2mesh.geometric import CameraMatrix


class KinectSensor:
    """The Kinect RGB-D sensor used in the NYU v2 RGB-D dataset and the TUM RGB-D dataset."""

    @staticmethod
    def get_camera_matrix() -> CameraMatrix:
        return CameraMatrix(fx=580., fy=580., cx=319.5, cy=239.5, width=640, height=480)
