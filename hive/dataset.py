#  HIVE, creates 3D mesh videos.
#  Copyright (C) 2024 Anthony Dickson anthony.dickson9656@gmail.com
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

import json
import os
from pathlib import Path
from typing import Dict, Tuple

import cv2
import numpy as np


class SyncTableData:
    def __init__(self, data: dict):
        self.index = np.asarray(data['index'], dtype=np.uint16)
        self.universal_time = np.asarray(data['univ_time'], dtype=np.float32)


class SyncTable:
    def __init__(self, path: str):
        with open(path, 'r') as f:
            data = json.load(f)

        self.vga = SyncTableData(data['vga'])
        self.hd = SyncTableData(data['hd'])


class KinectSyncTableNode:
    def __init__(self, data: dict):
        self.data: Dict[str, SyncTableData] = {node_name: SyncTableData(node_data) for node_name, node_data in data.items()}

    def __getitem__(self, item: str) -> SyncTableData:
        return self.data[item]


class KinectSyncTable:
    def __init__(self, path: str):
        with open(path, 'r') as f:
            data = json.load(f)

        kinect_data = data['kinect']
        self.color_data = KinectSyncTableNode(kinect_data['color'])
        self.depth_data = KinectSyncTableNode(kinect_data['depth'])


class KinectCalibrationNode:
    def __init__(self, data: dict):
        self.K_depth = np.asarray(data['K_depth'], dtype=np.float32)
        self.M_depth = np.asarray(data['M_depth'], dtype=np.float32)
        self.dist_coefficients_depth = np.asarray(data['distCoeffs_depth'], dtype=np.float32)
        self.depth_width = int(data['depth_width'])
        self.depth_height = int(data['depth_height'])

        self.K_color = np.asarray(data['K_color'], dtype=np.float32)
        self.M_color = np.asarray(data['M_color'], dtype=np.float32)
        self.dist_coefficients_color = np.asarray(data['distCoeffs_color'], dtype=np.float32)
        self.color_width = int(data['color_width'])
        self.color_height = int(data['color_height'])

        self.color_time_offset = int(data['color_time_offset'])
        self.depth_time_offset = int(data['depth_time_offset'])
        self.M_world2sensor = np.asarray(data['M_world2sensor'], dtype=np.int8)
        self.domeCenter = np.asarray(data['domeCenter'], dtype=np.float32)


class KinectCalibration:
    def __init__(self, data: dict):
        self.calibDataSource = data['calibDataSource']
        self.panopticCalibDataSource = data['panopticCalibDataSource']
        self.sensors = {i + 1: KinectCalibrationNode(sensor_data) for i, sensor_data in enumerate(data['sensors'])}
        self.M_world2vga = data['M_world2vga']

    def __getitem__(self, kinect_node: int) -> KinectCalibrationNode:
        return self.sensors[kinect_node]


class PanopticCamera:
    def __init__(self, data: dict):
        self.name = str(data['name'])
        self.type = str(data['type'])
        self.resolution = (int(data['resolution'][1]), int(data['resolution'][0]))  # height, width
        self.panel = int(data['panel'])
        self.node = int(data['node'])
        self.K = np.asarray(data['K'], dtype=np.float32)
        self.dist_coefficient = np.asarray(data['distCoef'], dtype=np.float32)
        self.R = np.asarray(data['R'], dtype=np.float32)
        self.t = np.asarray(data['t'], dtype=np.float32)


class PanopticCalibration:
    def __init__(self, data: dict):
        self.calib_data_source = data['calibDataSource']
        self.cameras = {camera_data['name']: PanopticCamera(camera_data) for camera_data in data['cameras']}

    def __getitem__(self, node_name: str) -> PanopticCamera:
        return self.cameras[node_name]


class CMUPanopticDataset:
    """
    Loader for CMU Panoptic Dataset http://domedb.perception.cs.cmu.edu/index.html.

    More specifically, this loader expects that:
    (1) the dataset has Kinect sensor data,
    (2) the dataset was downloaded with the script `getData_kinoptic.sh`,
    (3) the frame data was extracted with the script `hdImgsExtractor.sh`.
    """

    depth_parent_folder = 'kinect_shared_depth'
    depth_node_formatter = 'KINECTNODE{:d}'.format
    depth_filename = 'depthdata.dat'
    depth_to_meters = 1 / 1000
    depth_frame_width = 512
    depth_frame_height = 424
    depth_data_type = np.uint16
    depth_bytes_per_pixel = 2

    image_folder = 'kinectImgs'
    image_node_formatter = '50_{:02d}'.format
    image_filename_formatter = '50_{:02d}_{:08d}.jpg'.format

    calibration_filename_formatter = 'calibration_{}.json'.format
    kinect_calibration_filename_formatter = 'kcalibration_{}.json'.format

    sync_tables_filename_formatter = 'synctables_{}.json'.format
    kinect_sync_tables_filename_formatter = 'ksynctables_{}.json'.format

    def __init__(self, base_path: str):
        self.base_path = base_path
        self.dataset_name = Path(base_path).name

        self.sync_table = self.load_sync_table()
        self.kinect_sync_table = self.load_kinect_sync_table()
        self.camera_calibration = self.load_camera_calibration()
        self.kinect_calibration = self.load_kinect_calibration()

    def load_sync_table(self):
        path = os.path.join(self.base_path, self.sync_tables_filename_formatter(self.dataset_name))

        return SyncTable(path)

    def load_kinect_sync_table(self):
        path = os.path.join(self.base_path, self.kinect_sync_tables_filename_formatter(self.dataset_name))

        return KinectSyncTable(path)

    def load_camera_calibration(self):
        calibration_filename = self.calibration_filename_formatter(self.dataset_name)
        calibration_path = os.path.join(self.base_path, calibration_filename)

        with open(calibration_path, 'r') as f:
            camera_calibration = json.load(f)

        return PanopticCalibration(camera_calibration)

    def load_kinect_calibration(self):
        calibration_filename = self.kinect_calibration_filename_formatter(self.dataset_name)
        calibration_path = os.path.join(self.base_path, calibration_filename)

        with open(calibration_path, 'r') as f:
            camera_calibration = json.load(f)

        return KinectCalibration(camera_calibration)

    def get_image_path(self, kinect_node: int, index: int) -> str:
        """
        Get the path to the image for the given Kinect node and frame index.
        :param kinect_node: The Kinect node as an integer [1-10].
        :param index: The frame index to load [0, N).
        :return:
        """
        if kinect_node < 1 or kinect_node > 10:
            raise ValueError(f"Kinect node must be an integer between 1 and 10 (inclusive).")

        return os.path.join(self.base_path, self.image_folder, self.image_node_formatter(kinect_node),
                            self.image_filename_formatter(kinect_node, index))

    def get_image(self, kinect_node: int, index: int) -> np.ndarray:
        """
        Load an image from a Kinect node.
        :param kinect_node: The Kinect node as an integer [1-10].
        :param index: The frame index to load [0, N).
        :return: An RGB image (height, width, channels).
        """
        image = cv2.imread(self.get_image_path(kinect_node, index))

        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    def get_depth_path(self, kinect_node: int) -> str:
        """
        Get the path to the depth data for a given Kinect node.
        :param kinect_node: The Kinect node as an integer [1-10].
        :return: A path to depth data.
        """
        if kinect_node < 1 or kinect_node > 10:
            raise ValueError(f"Kinect node must be an integer between 1 and 10 (inclusive).")

        return os.path.join(self.base_path, self.depth_parent_folder, self.depth_node_formatter(kinect_node),
                            self.depth_filename)

    def get_depth_map(self, kinect_node: int, index: int) -> np.ndarray:
        """
        Load a depth map from a Kinect node.
        :param kinect_node: The Kinect node as an integer [1-10].
        :param index: The frame index to load [0, N).
        :return: A depth map in meters (float32).
        """
        bytes_per_frame = self.depth_bytes_per_pixel * self.depth_frame_height * self.depth_frame_width

        start_index = index * bytes_per_frame

        path = self.get_depth_path(kinect_node)

        with open(path, 'rb') as f:
            f.seek(start_index)
            byte_data = f.read(bytes_per_frame)

        flat_depth_map = np.frombuffer(byte_data, dtype=self.depth_data_type)
        depth_map = flat_depth_map.reshape((self.depth_frame_height, self.depth_frame_width))

        return depth_map * self.depth_to_meters

    def get_synced_frame_data(self, frame_index: int, kinect_node: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get the synchronised frame data for a given Kinect node.

        :param frame_index: The zero-based index of the frame data to retrieve.
        :param kinect_node: The one-based index of the Kinect node to use.
        :return: A 2-tuple containing the colour frame and depth map, in that order.
        """
        universal_time = self.sync_table.hd.universal_time[frame_index]
        node_name = self.depth_node_formatter(kinect_node)

        color_sync_table = self.kinect_sync_table.color_data[node_name]
        color_index = np.argmin(np.abs(universal_time - color_sync_table.universal_time - 6.25))
        color_time_distance = abs(universal_time - color_sync_table.universal_time[color_index])

        depth_sync_table = self.kinect_sync_table.depth_data[node_name]
        depth_index = np.argmin(np.abs(universal_time - depth_sync_table.universal_time))
        depth_time_distance = abs(universal_time - depth_sync_table.universal_time[depth_index])

        depth_color_diff = abs(depth_sync_table.universal_time[depth_index] -
                               color_sync_table.universal_time[color_index])

        if depth_color_diff > 6.25:
            raise RuntimeError(f"Kinect frame data for frame index {frame_index:,d} has a time difference of "
                               f"{depth_color_diff:,.2f}.")

        if color_time_distance > 30 or depth_time_distance > 17:
            raise RuntimeError(f"Kinect frame data for frame index {frame_index:,d} are too far apart.")

        color_frame = self.get_image(index=frame_index, kinect_node=kinect_node)
        depth_frame = self.get_depth_map(index=frame_index, kinect_node=kinect_node)

        return color_frame, depth_frame

    def kinect_to_world_coordinates(self, kinect_node: int):
        """
        Get the transformation matrix that transforms 3D coordinates from the local Kinect node space to world space.
        :param kinect_node: The index of the Kinect node (one-based).
        :return: A homogeneous 4x4 transformation matrix.
        """
        image_node_name = self.image_node_formatter(kinect_node)
        panoptic_calibration = self.camera_calibration[image_node_name]
        kinect_calibration = self.kinect_calibration[kinect_node]

        M = np.hstack((panoptic_calibration.R, panoptic_calibration.t))
        T_world_to_kinect = np.eye(4, dtype=np.float32)
        T_world_to_kinect[:3, :] = M
        T_kinect_color_to_panoptic_world = np.linalg.inv(T_world_to_kinect)

        scale_factor = 100  # cm to meter
        scale_kinoptic_to_panoptic = np.eye(4)
        scale_kinoptic_to_panoptic[0:2, 0:2] = scale_factor * scale_kinoptic_to_panoptic[0:2, 0:2]

        T_kinect_color_to_kinect_local = kinect_calibration.M_color
        T_kinect_local_to_kinect_color = np.linalg.inv(T_kinect_color_to_kinect_local)

        T_kinect_local_to_panoptic_world = T_kinect_color_to_panoptic_world * scale_kinoptic_to_panoptic * T_kinect_local_to_kinect_color

        return T_kinect_local_to_panoptic_world
