"""This module defines the camera parameters for common cameras/sensors."""

#  HIVE, creates 3D mesh videos.
#  Copyright (C) 2023 Anthony Dickson anthony.dickson9656@gmail.com
#
#  This program is free software: you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License
#  along with this program.  If not, see <https://www.gnu.org/licenses/>.

from hive.geometric import CameraMatrix


class KinectSensor:
    """The Kinect RGB-D sensor used in the NYU v2 RGB-D dataset and the TUM RGB-D dataset."""

    @staticmethod
    def get_camera_matrix() -> CameraMatrix:
        return CameraMatrix(fx=580., fy=580., cx=319.5, cy=239.5, width=640, height=480)
