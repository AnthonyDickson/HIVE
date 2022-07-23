"""
This module contains the code for converting datasets of various formats into the standardised format. This also
includes estimating camera parameters and depth maps (when specified).
"""

import contextlib
import enum
import logging
import os
import shutil
import subprocess
from abc import ABC
from os.path import join as pjoin
from pathlib import Path
from typing import Optional, Union, Tuple, List

import cv2
import imageio
import numpy as np
import torch
from scipy.interpolate import interp1d
from scipy.spatial.transform import Slerp, Rotation
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from thirdparty.dpt import dpt
from video2mesh.geometry import Trajectory
from video2mesh.image_processing import calculate_target_resolution
from video2mesh.io import DatasetBase, DatasetMetadata, VTMDataset, COLMAPProcessor, ImageFolderDataset, \
    create_masks, VideoMetadata, InvalidDatasetFormatError
from video2mesh.options import COLMAPOptions, BackgroundMeshOptions, StorageOptions, PipelineOptions
from video2mesh.types import Size, File
from video2mesh.utils import tqdm_imap


# TODO: Make depth estimation customisable via cli (e.g. max depth)


class DatasetAdaptor(DatasetBase, ABC):
    """Creates a copy of a dataset in the VTMDataset format."""

    def __init__(self, base_path: File, output_path: File, overwrite_ok=False, num_frames=-1, frame_step=1,
                 colmap_options=COLMAPOptions()):
        """
        :param base_path: The path to the dataset.
        :param output_path: The path to write the new dataset to.
        :param overwrite_ok: Whether it is okay to overwrite existing adapted dataset.
        :param num_frames: The maximum of frames to process. Set to -1 (default) to process all frames.
        :param frame_step: The frequency to sample frames at for COLMAP and pose optimisation.
            If set to 1, samples all frames (i.e. no effect). Otherwise if set to n > 1, samples every n frames.
        :param colmap_options: The configuration to use for COLMAP if estimating camera parameters.
        """
        super().__init__(base_path=base_path, overwrite_ok=overwrite_ok)

        self.output_path = output_path
        self.num_frames = num_frames
        self.frame_step = frame_step
        self.colmap_options = colmap_options

    def get_full_num_frames(self) -> int:
        """The number of frames in the non-truncated dataset."""
        raise NotImplementedError

    def get_metadata(self, estimate_pose: bool, estimate_depth: bool) -> DatasetMetadata:
        """
        Get the metadata object for this dataset.

        :param estimate_pose: Whether to estimate camera parameters with COLMAP or use provided ground truth
            camera parameters.
        :param estimate_depth: Whether to estimate depth maps or use provided ground truth depth maps.
        """
        raise NotImplementedError

    def get_camera_trajectory(self) -> Trajectory:
        """
        Get the ground truth camera trajectory.
        If the dataset does not have a ground truth camera trajectory, a `NotImplementedError` is raised.

        :return: The (N, 7) ground truth camera trajectory, if it exists.
        """
        trajectory = np.vstack([self.get_pose(i) for i in range(self.num_frames)])
        trajectory = Trajectory(trajectory)

        return trajectory

    def get_camera_matrix(self) -> np.ndarray:
        """
        Get the ground truth camera intrinsic matrix.
        If the dataset does not have ground truth camera parameters, a `NotImplementedError` is raised.

        :return: The camera intrinsic parameters (3, 3) matrix, if it exists.
        """
        raise NotImplementedError

    def get_pose(self, index: int) -> np.ndarray:
        """
        Get the ground truth pose for the frame at `index`.
        If the dataset does not have ground truth pose data, a `NotImplementedError` is raised.

        :param index: The zero-based frame index of the pose to fetch.
        :return: The pose (quaternion 4-vector, position 3-vector) at frame `index`, if it exists.
        """
        raise NotImplementedError

    def get_frame(self, index: int) -> np.ndarray:
        """
        Get the RGB frame at `index`.

        :param index: The zero-based index of the frame to fetch.
        :return: The frame at `index`.
        """
        raise NotImplementedError

    def get_depth_map(self, index: int) -> np.ndarray:
        """
        Get the ground truth depth map for the frame at `index`.
        If the dataset does not have ground truth depth maps, a `NotImplementedError` is raised.

        :param index: The zero-based frame index of the depth map to fetch.
        :return: The depth map at frame `index`, if it exists.
        """
        raise NotImplementedError

    def copy_frames(self, output_path: str, num_frames=-1):
        """
        Copy frames to the specified folder.
        This may be faster than calling `.get_frame(i)` followed by writing the frame to the destination folder.

        :param output_path: The folder to save the frames to.
        :param num_frames: How many frames to copy. If set to -1, copy all frames.
        """
        num_frames = self.num_frames if num_frames == -1 else num_frames

        def copy_image(index: int):
            image = self.get_frame(index)
            output_image_path = pjoin(output_path, VTMDataset.index_to_filename(index))
            imageio.v3.imwrite(output_image_path, image)

        tqdm_imap(copy_image, range(num_frames))

    def copy_depth_maps(self, output_path: str):
        """
        Copy depth maps to a folder.
        This may be faster than calling `.get_depth_map(i)` followed by writing the depth map to the destination folder.

        :param output_path: The folder to save the depth maps to.
        """

        def copy_image(index: int):
            image = self.get_depth_map(index)
            output_image_path = pjoin(output_path, VTMDataset.index_to_filename(index))
            imageio.v3.imwrite(output_image_path, image)

        tqdm_imap(copy_image, range(self.num_frames))

    def convert_from_ground_truth(self) -> VTMDataset:
        """
        Read the dataset and create a copy in the VTMDataset format.

        :return: The newly created dataset object.
        """
        return self.convert(estimate_pose=True, estimate_depth=True)

    def convert_from_rgb(self) -> VTMDataset:
        """
        Read a video file and create a VTMDataset with estimated camera parameters and depth maps.

        :return: The newly created dataset object.
        """
        # TODO: Add way to pass arguments to PoseOptimiser object.
        return self.convert(estimate_pose=False, estimate_depth=False)

    def convert(self, estimate_pose: bool, estimate_depth: bool) -> VTMDataset:
        """
        Convert a dataset into the standard format.

        :param estimate_pose: Whether to estimate camera parameters with COLMAP or use provided ground truth
            camera parameters.
        :param estimate_depth: Whether to estimate depth maps or use provided ground truth depth maps.
        :return: The converted dataset.
        """
        if cached_dataset := self._try_get_cached_dataset(estimate_pose=estimate_pose, estimate_depth=estimate_depth):
            logging.info(f"Found cached dataset at {self.output_path}.")
            return cached_dataset

        output_image_folder, output_depth_folder, output_mask_folder = self._setup_folders()

        logging.info(f"Creating metadata for dataset.")
        metadata = self.get_metadata(estimate_pose, estimate_depth)
        metadata_path = pjoin(self.output_path, VTMDataset.metadata_filename)
        metadata.save(metadata_path)

        logging.info(f"Copying RGB frames.")
        self.copy_frames(output_image_folder)

        create_masks(
            DataLoader(ImageFolderDataset(output_image_folder), batch_size=8),
            mask_folder=output_mask_folder,
            overwrite_ok=True
        )

        if estimate_depth:
            logging.info(f"Creating depth maps.")
            estimate_depth_dpt(ImageFolderDataset(output_image_folder), output_depth_folder)
        else:
            logging.info(f"Copying depth maps.")
            self.copy_depth_maps(output_depth_folder)

        if estimate_pose:
            debug_folder = pjoin(self.output_path, 'debug')
            camera_matrix, camera_trajectory = self._estimate_camera_parameters(debug_folder, output_depth_folder,
                                                                                metadata)
        else:
            camera_matrix = self.get_camera_matrix()
            camera_trajectory = self.get_camera_trajectory()

        logging.info(f"Creating camera matrix file.")
        camera_matrix_path = pjoin(self.output_path, VTMDataset.camera_matrix_filename)
        # noinspection PyTypeChecker
        np.savetxt(camera_matrix_path, camera_matrix)

        logging.info(f"Creating camera trajectory file.")
        camera_trajectory_path = pjoin(self.output_path, VTMDataset.camera_trajectory_filename)
        # noinspection PyTypeChecker
        camera_trajectory.save(camera_trajectory_path)

        logging.info(f"Created new dataset at {self.output_path}.")

        return VTMDataset(self.output_path, overwrite_ok=self.overwrite_ok)

    def _try_get_cached_dataset(self, estimate_pose: bool, estimate_depth: bool) -> Optional[VTMDataset]:
        """
        Attempt to load a cached dataset.

        If there is a correctly formatted VTMDataset that was created with the exact same configuration, then that
        will be returned.

        :param estimate_pose: Whether to estimate camera parameters with COLMAP or use provided ground truth
            camera parameters.
        :param estimate_depth: Whether to estimate depth maps or use provided ground truth depth maps.
        :return: The cached dataset, if it exists. Otherwise, returns `None`.
        """
        if VTMDataset.is_valid_folder_structure(self.output_path):
            dataset = VTMDataset(self.output_path, overwrite_ok=self.overwrite_ok)

            num_frames = len(os.listdir(dataset.path_to_rgb_frames))
            num_depth_maps = len(os.listdir(dataset.path_to_depth_maps))

            same_num_frames = (num_frames == self.num_frames) or (num_depth_maps == self.num_frames)

            expected_camera_trajectory_length = self.num_frames
            camera_trajectory_length = len(dataset.camera_trajectory)

            same_trajectory_length = expected_camera_trajectory_length == camera_trajectory_length
            # TODO: Check whether the cached dataset was created with the same settings
            #  (e.g. GT vs estimated data, frame step).

            same_metadata = dataset.metadata == self.get_metadata(estimate_pose, estimate_depth)

            if same_num_frames and same_trajectory_length and same_metadata:
                return dataset

        return None

    def _delete_cache(self):
        if os.path.isdir(self.output_path):
            if self.overwrite_ok:
                logging.warning(f"Found a dataset in {self.output_path} but it was created with different settings."
                                f"Since `overwrite_ok` is set to `True`, the existing dataset will be deleted.")
                shutil.rmtree(self.output_path)
            else:
                raise RuntimeError(f"The output path {self.output_path} already exists! "
                                   f"Possible fix: Delete or rename the file/folder.")

    def _setup_folders(self) -> Tuple[str, str, str]:
        """
        Create the output folders.

        :return: The paths to the: image folder, depth map folder and instance segmentation masks folder.
        """
        self._delete_cache()
        os.makedirs(self.output_path, exist_ok=self.overwrite_ok)

        image_folder = create_folder(self.output_path, VTMDataset.rgb_folder, exist_ok=self.overwrite_ok)
        depth_folder = create_folder(self.output_path, VTMDataset.depth_folder, exist_ok=self.overwrite_ok)
        mask_folder = create_folder(self.output_path, VTMDataset.mask_folder, exist_ok=self.overwrite_ok)

        return image_folder, depth_folder, mask_folder

    @staticmethod
    def _get_frame_subset(num_frames, frame_step):
        """
        Get frame indices for the full set and the subset using `frame_step`.

        :param num_frames: How many frames to include.
        :param frame_step: The step between frames.

        :return: The list of indices and the list of indices
        """
        frames = list(range(num_frames))
        frames_subset = frames[::frame_step]

        if frames_subset[-1] != frames[-1]:
            frames_subset += [frames[-1]]

        return frames, frames_subset

    def _estimate_camera_parameters(self, output_folder: str, output_depth_folder: str, metadata: DatasetMetadata) -> \
            Tuple[np.ndarray, Trajectory]:
        """
        Estimate the camera parameters (intrinsics and extrinsics) with COLMAP.

        :param output_folder: The folder to save the COLMAP output to.
        :param output_depth_folder: Where the estimated depth maps have been saved to.
        :param metadata: The metadata for the dataset.
        :return: The 3x3 camera matrix and the Nx7 camera poses.
        """
        colmap_log_file = pjoin(output_folder, 'colmap_logs.txt')

        if os.path.exists(output_folder):
            shutil.rmtree(output_folder)

        colmap_folder = create_folder(output_folder, 'colmap')
        colmap_rgb_path = create_folder(colmap_folder, 'rgb')
        colmap_workspace_path = create_folder(colmap_folder, 'workspace')

        frames, frames_subset = self._get_frame_subset(self.get_full_num_frames(), self.frame_step)

        logging.info("Copying RGB frames for COLMAP...")

        self.copy_frames(colmap_rgb_path, self.get_full_num_frames())

        if self.frame_step > 1:
            for index in set(frames).difference(frames_subset):
                os.remove(pjoin(colmap_rgb_path, VTMDataset.index_to_filename(index)))

            for dst_index, src_index in enumerate(frames_subset):
                src_path = pjoin(colmap_rgb_path, VTMDataset.index_to_filename(src_index))
                dst_path = pjoin(colmap_rgb_path, VTMDataset.index_to_filename(dst_index))

                shutil.move(src_path, dst_path)

        logging.info(f"Running COLMAP... This might take a while!")
        logging.info(f"Check {colmap_log_file} for the logs.")

        colmap_processor = COLMAPProcessor(image_path=colmap_rgb_path, workspace_path=colmap_workspace_path,
                                           colmap_options=self.colmap_options)

        with open(colmap_log_file, 'w') as f, contextlib.redirect_stdout(f), contextlib.redirect_stderr(f):
            colmap_processor.run()

        camera_matrix, camera_poses_scaled = self._get_scaled_colmap_camera_params(colmap_processor,
                                                                                   output_depth_folder,
                                                                                   metadata, frames_subset)

        if self.frame_step > 1:
            camera_poses_scaled = self._interpolate_poses(camera_poses_scaled, keyframes=frames_subset)

        camera_poses_scaled = Trajectory(camera_poses_scaled[:self.num_frames])

        return camera_matrix, camera_poses_scaled.normalise()

    @staticmethod
    def _get_scaled_colmap_camera_params(colmap_processor: COLMAPProcessor,
                                         output_depth_folder: str,
                                         metadata: DatasetMetadata,
                                         frames_subset: List[int]) -> Tuple[np.ndarray, Trajectory]:
        """
        Scale the camera poses estimated by COLMAP to match the scale in the estimated depth maps.

        :param colmap_processor: The COLMAP processor object.
        :param output_depth_folder: The path to the estimated depth maps.
        :param metadata: The dataset metadata.
        :param frames_subset: The subset of frames that were used for COLMAP.

        :return: The estimated 3x3 camera matrix and the scaled Nx7 camera poses.
        """
        logging.info("Scaling COLMAP Poses...")

        logging.info("Creating COLMAP depth maps...")
        camera_matrix, camera_poses = colmap_processor.load_camera_params(raw_pose=True)

        if colmap_processor.colmap_options.dense:
            colmap_depth = colmap_processor.get_dense_depth_maps(resize_to=(metadata.height, metadata.width))
        else:
            colmap_depth = colmap_processor.get_sparse_depth_maps(camera_matrix, camera_poses)

        def transform(depth_map):
            depth_map = VTMDataset.depth_scaling_factor * depth_map.astype(np.float32)
            depth_map[depth_map > metadata.max_depth] = 0.0

            return depth_map

        logging.info("\tLoading estimated depth maps...")
        depth_dataset = ImageFolderDataset(output_depth_folder, transform=transform)
        depth_frame_subset = list(filter(lambda index: index < len(depth_dataset), frames_subset))
        est_depth = np.asarray(tqdm_imap(depth_dataset.__getitem__, depth_frame_subset))

        num_frames = min(len(colmap_depth), len(est_depth))

        colmap_depth = colmap_depth[:num_frames]
        est_depth = est_depth[:num_frames]
        nonzero_mask = (colmap_depth > 0.) & (est_depth > 0.)

        # Find scale which when applied to the colmap poses, minimises the loss function below.
        scaling_factor = np.median(est_depth[nonzero_mask] / colmap_depth[nonzero_mask])
        colmap_depth_scaled = scaling_factor * colmap_depth

        def loss(pred_depth, gt_depth):
            return np.exp(np.median(np.log(1. / pred_depth) - np.log(1. / gt_depth))) - 1.0

        sigma_before = loss(est_depth[nonzero_mask], colmap_depth[nonzero_mask])
        sigma_after = loss(est_depth[nonzero_mask], colmap_depth_scaled[nonzero_mask])

        logging.info(f"Depth Scale: {scaling_factor:.4f} - Loss Before: {sigma_before:.4f} - "
                     f"Loss After: {sigma_after:.4f}")

        camera_poses_scaled = camera_poses.copy()
        camera_poses_scaled[:, 4:] *= scaling_factor

        # TODO: Integrate COMAP depth maps more closely into the VTMDataset format.
        # TODO: Fix edge case where last frame index is not multiple of frame step and is not copied
        #  (See how frame_subset is created).
        if colmap_processor.colmap_options.dense:
            parent_path = Path(output_depth_folder).parent
            colmap_depth_output_path = pjoin(parent_path, 'colmap_depth')
            os.makedirs(colmap_depth_output_path)

            def save_depth(index_depth_map):
                index, depth_map = index_depth_map
                depth_map = 1000 * depth_map  # convert to mm
                depth_map = depth_map.astype(np.uint16)
                imageio.v3.imwrite(pjoin(colmap_depth_output_path, VTMDataset.index_to_filename(index)), depth_map)

            logging.debug(f"Writing dense COLMAP depth maps to {colmap_depth_output_path}...")
            tqdm_imap(save_depth, list(zip(frames_subset, colmap_depth_scaled)))

        return camera_matrix, camera_poses_scaled

    @staticmethod
    def _interpolate_poses(poses: Trajectory, keyframes: List[int]) -> Trajectory:
        """
        Interpolate the pose for frames with missing data (when `frame_step` > 1).

        :param poses: The poses estimated by COLMAP over a subset of frames.
        :param keyframes: The indices of the frames COLMAP was run over.
        :return: The interpolated Nx7 camera poses.
        """
        src_num_frames = len(poses)
        dst_num_frames = max(keyframes) + 1

        interpolated_poses = np.zeros((dst_num_frames, poses.shape[1]))

        src_indices = zip(range(src_num_frames - 1), range(1, src_num_frames))
        dst_indices = zip(keyframes[:-1], keyframes[1:])

        for (src_start, src_end), (dst_start, dst_end) in zip(src_indices, dst_indices):
            start_rotation = poses[src_start, :4]
            end_rotation = poses[src_end, :4]
            start_position = poses[src_start, 4:]
            end_position = poses[src_end, 4:]

            key_frame_times = [0, 1]
            times_to_interpolate = np.linspace(0, 1, num=(dst_end + 1) - dst_start)

            slerp = Slerp(times=key_frame_times, rotations=Rotation.from_quat([start_rotation, end_rotation]))
            lerp = interp1d(key_frame_times, [start_position, end_position], axis=0)

            interpolated_poses[dst_start:dst_end + 1, 4:] = lerp(times_to_interpolate)
            interpolated_poses[dst_start:dst_end + 1, :4] = slerp(times_to_interpolate).as_quat()

        return Trajectory(interpolated_poses)


class TUMAdaptor(DatasetAdaptor):
    """
    Converts image, depth and pose data from a TUM formatted dataset to the VTM dataset format.
    """
    # The below values are fixed and common to all subsets of the TUM dataset.
    fx = 525.0  # focal length x
    fy = 525.0  # focal length y
    cx = 319.5  # optical center x
    cy = 239.5  # optical center y
    width = 640
    height = 480
    intrinsic_matrix = np.array([[fx, 0., cx],
                                 [0., fy, cy],
                                 [0., 0., 1.]])

    fps = 30.0
    frame_time = 1.0 / fps

    """The name/path of the file that contains the camera pose information."""
    pose_path = "groundtruth.txt"

    """The name/path of the file that contains the mapping of timestamps to image file paths."""
    rgb_files_path = "rgb.txt"

    """The name/path of the file that contains the mapping of timestamps to depth map paths."""
    depth_map_files_path = "depth.txt"

    required_files = [pose_path, rgb_files_path, depth_map_files_path]

    rgb_folder = "rgb"
    depth_folder = "depth"
    required_folders = [rgb_folder, depth_folder]

    def __init__(self, base_path: File, output_path: File, overwrite_ok=False, num_frames=-1, frame_step=1,
                 colmap_options=COLMAPOptions(), is_16_bit=True):
        """
        :param base_path: The path to the dataset.
        :param output_path: The path to write the new dataset to.
        :param overwrite_ok: Whether it is okay to overwrite existing adapted dataset.
        :param num_frames: The maximum of frames to process. Set to -1 (default) to process all frames.
        :param frame_step: The frequency to sample frames at for COLMAP and pose optimisation.
            If set to 1, samples all frames (i.e. no effect). Otherwise if set to n > 1, samples every n frames.
        :param colmap_options: The configuration to use for COLMAP if estimating camera parameters.
        :param is_16_bit: Whether the images are stored with 16-bit values or 32-bit values.
        """
        super().__init__(base_path=base_path, output_path=output_path, overwrite_ok=overwrite_ok, num_frames=num_frames,
                         frame_step=frame_step, colmap_options=colmap_options)

        self.base_path = Path(base_path)
        self.pose_path = Path(pjoin(base_path, str(Path(self.pose_path))))
        self.rgb_files_path = Path(pjoin(base_path, str(Path(self.rgb_files_path))))
        self.depth_map_files_path = Path(pjoin(base_path, str(Path(self.depth_map_files_path))))

        self.is_16_bit = is_16_bit
        # The depth maps need to be divided by 5000 for the 16-bit PNG files
        # or 1.0 (i.e. no effect) for the 32-bit float images in the ROS bag files
        self.depth_scale_factor = 1.0 / 5000.0 if is_16_bit else 1.0

        self.image_filenames, self.depth_filenames, self.camera_trajectory = self._get_synced_frame_data()

        if num_frames == -1:
            self.num_frames = len(self.image_filenames)

        # Take inverse since TUM poses are cam-to-world, but most of the math assumes world-to-cam poses.
        self.camera_trajectory = self.camera_trajectory.normalise().inverse()
        # TODO: Account for initial orientation of the kinect device?

    def _get_synced_frame_data(self):
        """
        Get the set of matching frames.
        The TUM dataset is created with a Kinect sensor.
        The colour images and depth maps given by this sensor are not synchronised and as such the timestamps never
        perfectly match.
        Therefore, we need to associate the frames with the closest timestamps to get the best set of frame pairs.

        :return: Three lists each containing: paths to the colour frames, paths to the depth maps and the camera poses.
        # return A list of 2-tuples each containing the paths to a colour image and depth map.
        """

        def load_timestamps_and_paths(list_path):
            timestamps = []
            data = []

            with open(str(list_path), 'r') as f:
                for line in f:
                    line = line.strip()

                    if line.startswith('#'):
                        continue

                    parts = line.split(' ')
                    timestamp = float(parts[0])
                    data_parts = parts[1:]

                    timestamps.append(timestamp)
                    data.append(data_parts)

            timestamps = np.array(timestamps)
            data = np.array(data)

            return timestamps, data

        image_timestamps, image_paths = load_timestamps_and_paths(self.rgb_files_path)
        depth_map_timestamps, depth_map_paths = load_timestamps_and_paths(self.depth_map_files_path)
        trajectory_timestamps, trajectory_data = load_timestamps_and_paths(self.pose_path)

        def get_match_indices(query, target):
            # This creates a M x N matrix of the difference between each of the image and depth map timestamp pairs
            # where M is the number of images and N is the number of depth maps.
            timestamp_deltas = np.abs(query.reshape(-1, 1) - target.reshape(1, -1))
            # There are more images than depth maps. So what we need is a 1:1 mapping from depth maps to images.
            # Taking argmin along the columns (axis=0) gives us index of the closest image timestamp for each
            # depth map timestamp.
            corresponding_indices = timestamp_deltas.argmin(axis=0)

            return corresponding_indices

        # Select the matching images.
        image_indices = get_match_indices(image_timestamps, depth_map_timestamps)
        image_filenames_subset = image_paths[image_indices]
        # data loaded by `load_timestamps_and_paths(...)` gives data as a 2d array (in this case a column vector),
        # but we want the paths as a 1d array.
        image_filenames_subset = image_filenames_subset.flatten()
        # Image filenames include the prefix `rgb/`, so we cut that part off.
        image_filenames_subset = map(lambda path: path[4:], image_filenames_subset)
        image_filenames_subset = list(image_filenames_subset)

        depth_map_subset = depth_map_paths.flatten()
        # Similar to the image filenames, the depth map filenames include the prefix `depth/`, so we cut that part off.
        depth_map_subset = map(lambda path: path[6:], depth_map_subset)
        depth_map_subset = list(depth_map_subset)

        # Select the matching trajectory readings.
        trajectory_indices = get_match_indices(trajectory_timestamps, depth_map_timestamps)
        trajectory_subset = trajectory_data[trajectory_indices]

        def process_trajectory_datum(datum):
            tx, ty, tz, qx, qy, qz, qw = map(float, datum)

            return qx, qy, qz, qw, tx, ty, tz

        trajectory_subset = np.array(list(map(process_trajectory_datum, trajectory_subset)))

        return image_filenames_subset, depth_map_subset, Trajectory(trajectory_subset)

    def get_frame_path(self, index) -> str:
        return pjoin(self.base_path, self.rgb_folder, self.image_filenames[index])

    def get_depth_map_path(self, index) -> str:
        return pjoin(self.base_path, self.depth_folder, self.depth_filenames[index])

    def get_full_num_frames(self):
        a, _, _ = self._get_synced_frame_data()
        return len(a)

    def get_metadata(self, estimate_pose: bool, estimate_depth: bool) -> DatasetMetadata:
        # TODO: Make the depth mask dilation iterations configurable.
        # This gets the default value for `depth_mask_dilation_iterations`.
        depth_mask_dilation_iterations = BackgroundMeshOptions().depth_mask_dilation_iterations

        return DatasetMetadata(num_frames=self.num_frames, frame_step=self.frame_step,
                               fps=self.fps, width=self.width, height=self.height,
                               estimate_pose=estimate_pose, estimate_depth=estimate_depth,
                               depth_mask_dilation_iterations=depth_mask_dilation_iterations,
                               depth_scale=VTMDataset.depth_scaling_factor, colmap_options=self.colmap_options)

    def get_camera_matrix(self) -> np.ndarray:
        return self.intrinsic_matrix

    def get_pose(self, index: int) -> np.ndarray:
        return self.camera_trajectory[index]

    def get_frame(self, index: int) -> np.ndarray:
        return imageio.v3.imread(self.get_frame_path(index))

    def get_depth_map(self, index: int) -> np.ndarray:
        depth_map = imageio.v3.imread(self.get_depth_map_path(index))
        depth_map = depth_map * self.depth_scale_factor  # convert to metres from non-standard scale & units.
        depth_map = (1000 * depth_map).astype(np.uint16)  # convert to mm from metres.

        return depth_map

    def copy_frames(self, output_path: str, num_frames=-1):
        num_frames = self.num_frames if num_frames == -1 else num_frames

        def copy_image(index: int):
            image_path = self.get_frame_path(index)
            output_image_path = pjoin(output_path, VTMDataset.index_to_filename(index))
            shutil.copy(image_path, output_image_path)

        tqdm_imap(copy_image, range(num_frames))


class VideoAdaptorBase(DatasetAdaptor, ABC):
    """Base class for adaptors of video datasets."""

    def __init__(self, base_path: File, output_path: File, video_path: Union[str, Path], overwrite_ok=False,
                 num_frames=-1, frame_step=1, colmap_options=COLMAPOptions(),
                 resize_to: Optional[Union[int, Size]] = None):
        """
        :param base_path: The folder containing the video file.
        :param video_path: The path to the RGB video.
        :param output_path: The path to write the new dataset to.
        :param overwrite_ok: Whether it is okay to overwrite existing adapted dataset.
        :param num_frames: The maximum of frames to process. Set to -1 (default) to process all frames.
        :param frame_step: The frequency to sample frames at for COLMAP and pose optimisation.
            If set to 1, samples all frames (i.e. no effect). Otherwise if set to n > 1, samples every n frames.
        :param colmap_options: The configuration to use for COLMAP if estimating camera parameters.
        :param resize_to: The resolution (height, width) to resize the images to.
            If an int is given, the longest side will be scaled to this value and the shorter side will have its new
            length automatically calculated.
        """
        super().__init__(base_path=base_path, output_path=output_path, overwrite_ok=overwrite_ok, num_frames=num_frames,
                         frame_step=frame_step, colmap_options=colmap_options)

        self.video_path = video_path
        self.full_num_frames = self._count_frames()
        self.num_frames = self.full_num_frames if self.num_frames == -1 else self.num_frames

        with self.open_video(self.video_path) as video:
            self.source_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.source_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

        if isinstance(resize_to, tuple):
            resize_width, resize_height = resize_to
            self.target_height, self.target_width = \
                calculate_target_resolution((self.source_height, self.source_width),
                                                 (resize_height, resize_width))
        elif isinstance(resize_to, int):
            # noinspection PyTypeChecker
            self.target_height, self.target_width = \
                calculate_target_resolution((self.source_height, self.source_width), resize_to)
        else:
            self.target_height, self.target_width = self.source_height, self.source_width

        if self.target_height != self.source_height or self.target_width != self.source_width:
            logging.info(f"Will resize frames from {self.source_width}x{self.source_height} to "
                         f"{self.target_width}x{self.target_height} (width, height).")

    def _count_frames(self) -> int:
        """
        Count the number of frames in the video sequence.
        **Note**: This is done by getting and decoding each frame which will be slow (somewhere up to 3ms per frame).
        Frames are counted this way since the metadata can be inaccurate, but this method is always accurate.

        :return: The number of frames in the video.
        """
        logging.debug(f"Counting frames for the video {self.video_path}, this may take a few seconds...")
        num_frames = 0

        with self.open_video(self.video_path) as video:
            while video.isOpened():
                has_frame = video.grab()

                if has_frame:
                    num_frames += 1
                else:
                    break

        return num_frames

    @staticmethod
    @contextlib.contextmanager
    def open_video(video_path):
        video = cv2.VideoCapture(video_path)

        try:
            yield video
        finally:
            if video.isOpened():
                video.release()

    def get_metadata(self, estimate_pose: bool, estimate_depth: bool) -> DatasetMetadata:
        with self.open_video(self.video_path) as video:
            fps = float(video.get(cv2.CAP_PROP_FPS))

        height, width = self.target_height, self.target_width

        video_metadata = VideoMetadata(self.video_path, width=width, height=height, num_frames=self.num_frames, fps=fps)

        # This gets the default value for `depth_mask_dilation_iterations`.
        depth_mask_dilation_iterations = BackgroundMeshOptions().depth_mask_dilation_iterations

        return DatasetMetadata(num_frames=video_metadata.num_frames, fps=video_metadata.fps, width=width, height=height,
                               frame_step=self.frame_step, estimate_pose=estimate_pose, estimate_depth=estimate_depth,
                               depth_mask_dilation_iterations=depth_mask_dilation_iterations,
                               depth_scale=VTMDataset.depth_scaling_factor, colmap_options=self.colmap_options)

    def get_full_num_frames(self):
        return self.full_num_frames

    def get_frame(self, index: int) -> np.ndarray:
        with self.open_video(self.video_path) as video:
            video.set(cv2.CAP_PROP_POS_FRAMES, index - 1)

            read_frame, frame = video.read()

        if read_frame:
            return cv2.cvtColor(cv2.resize(frame, (self.target_width, self.target_height)), cv2.COLOR_BGR2RGB)
        else:
            raise RuntimeError(f"Could not read frame {index} (zero-based index) from the video {self.video_path}.")

    def copy_frames(self, output_path: str, num_frames=-1):
        num_frames = self.num_frames if num_frames == -1 else num_frames

        self.extract_video(self.video_path, output_path, num_frames,
                           target_resolution=(self.target_height, self.target_width))

    @staticmethod
    def extract_video(path_to_video: str, output_path: str, num_frames: int = -1,
                      target_resolution: Optional[Tuple[int, int]] = None,
                      rotation: Optional[int] = None):
        """
        Extract the frames from a video file.

        :param path_to_video: The path to the video file.
        :param output_path: The folder to save the frames to.
        :param num_frames: The maximum number of frames to extract. If set to -1, all frames are extracted.
        :param target_resolution: (optional) The resolution (height, width) to resize frames to.
        :param rotation: (optional) The rotation to apply to the video frames (see `cv2.ROTATE_*`).
        """
        ffmpeg_command = ['ffmpeg', '-i', path_to_video]

        if num_frames != -1:
            ffmpeg_command += ['-frames:v', str(num_frames)]

        if target_resolution is not None:
            height, width = target_resolution
            ffmpeg_command += ['-s', f"{width}x{height}"]

        if rotation == cv2.ROTATE_90_CLOCKWISE:
            ffmpeg_command += ['-vf', "transpose=1"]
        elif rotation == cv2.ROTATE_180:
            ffmpeg_command += ['-vf', "transpose=1,transpose=1"]
        elif rotation == cv2.ROTATE_90_COUNTERCLOCKWISE:
            ffmpeg_command += ['-vf', "transpose=2"]
        elif rotation is not None:
            raise ValueError(f"Expected `rotation` to be one of the following values: "
                             f"[cv2.ROTATE_90_CLOCKWISE ({cv2.ROTATE_90_CLOCKWISE}), "
                             f"[cv2.ROTATE_180 ({cv2.ROTATE_180}), "
                             f"[cv2.ROTATE_90_COUNTERCLOCKWISE ({cv2.ROTATE_90_COUNTERCLOCKWISE})], "
                             f"but got {rotation} instead.")

        ffmpeg_command += ['-start_number', str(0), pjoin(output_path, '%06d.png')]

        process = subprocess.Popen(ffmpeg_command)

        if (return_code := process.wait()) != 0:
            raise RuntimeError(f"Something went wrong with ffmpeg (return code {return_code}), "
                               f"check the logs for more details.")


class VideoAdaptor(VideoAdaptorBase):
    """Converts an RGB video to the VTMDataset format."""

    def __init__(self, base_path: File, output_path: File, overwrite_ok=False, num_frames=-1, frame_step=1,
                 colmap_options=COLMAPOptions(),
                 resize_to: Optional[Union[int, Size]] = None):
        """
        :param base_path: The path to the video.
        :param output_path: The path to write the new dataset to.
        :param overwrite_ok: Whether it is okay to overwrite existing adapted dataset.
        :param num_frames: The maximum of frames to process. Set to -1 (default) to process all frames.
        :param frame_step: The frequency to sample frames at for COLMAP and pose optimisation.
            If set to 1, samples all frames (i.e. no effect). Otherwise if set to n > 1, samples every n frames.
        :param colmap_options: The configuration to use for COLMAP if estimating camera parameters.
        :param resize_to: The resolution (height, width) to resize the images to.
            If an int is given, the longest side will be scaled to this value and the shorter side will have its new
            length automatically calculated.
        """
        path = Path(base_path)
        base_path = path.parent
        video_filename = path.name
        video_path = pjoin(base_path, video_filename)

        super().__init__(base_path=base_path, output_path=output_path, video_path=video_path, num_frames=num_frames,
                         frame_step=frame_step, overwrite_ok=overwrite_ok, colmap_options=colmap_options,
                         resize_to=resize_to)

    @classmethod
    def _validate_dataset(cls, base_path):
        """
        Check whether the given path points to a valid RGB-D dataset.

        This method will throw an AssertionError if the path does not point to a valid dataset.

        :param base_path: The path to the RGB-D dataset.
        """
        if os.path.isfile(base_path) and Path(base_path).suffix == '.mp4':
            return
        if os.path.isdir(base_path):
            items = os.listdir(base_path)
            paths = [os.path.join(base_path, item) for item in items]
            files = list(filter(os.path.isfile, paths))

            if len(files) < 1:
                raise InvalidDatasetFormatError(f"The folder {base_path} contains no files.")
            else:
                for file in files:
                    if Path(file).suffix == '.mp4':
                        break
                else:
                    raise InvalidDatasetFormatError(f"Could not find a '.mp4' video file in the folder {base_path}.")
        else:
            raise InvalidDatasetFormatError(f"The folder {base_path} does not exist!")

    def _load_camera_trajectory(self):
        raise NotImplementedError

    def get_camera_matrix(self) -> np.ndarray:
        raise NotImplementedError

    def get_pose(self, index: int) -> np.ndarray:
        raise NotImplementedError

    def get_camera_trajectory(self) -> np.ndarray:
        raise NotImplementedError

    def get_depth_map(self, index: int) -> np.ndarray:
        raise NotImplementedError

    def convert_from_ground_truth(self) -> VTMDataset:
        raise NotImplementedError


# noinspection PyArgumentList
class DeviceOrientation(enum.Enum):
    """Names associating device orientation to a cardinal direction.
    Rotation axis is about the z-axis, going counter-clockwise, with zero pointing in the negative x-axis (left).
    """
    # Lock button up (iPhone 13), no rotation
    Landscape = enum.auto()
    # 'Natural' vertical orientation, front-facing camera at top, 90 degrees CW
    Portrait = enum.auto()
    # Volume buttons up (iPhone 13), 180 degrees CW or CCW
    LandscapeReverse = enum.auto()
    # Upside down, 90 degrees CCW
    PortraitReverse = enum.auto()

    @classmethod
    def from_angle(cls, angle, degrees=False) -> 'DeviceOrientation':
        """
        Get device orientation from an angle (roll).

        :param angle: An angle between -180 and 180 [-PI, PI], usually the rotation about the z-axis (roll).
        :param degrees: Whether the given angle is in degrees.
        :return: The device orientation
        """
        if not degrees:
            angle = np.rad2deg(angle)

        # Divide circle into four quadrants with each quadrant extending 45 degrees either side of each
        # cardinal direction (0, 90, 190 and 270 degrees).
        if abs(angle) <= 45:
            return DeviceOrientation.Landscape
        elif -135 <= angle < -45:
            return DeviceOrientation.Portrait
        elif 45 < angle <= 135:
            return DeviceOrientation.PortraitReverse
        elif 135 < abs(angle) <= 180:
            return DeviceOrientation.LandscapeReverse
        else:
            angle_message = f"Expected angle in interval [-180, 180], got {angle}"

            if degrees:
                exception_message = f"{angle_message}."
            else:
                exception_message = f"{angle_message} (angle converted from radians)."

            raise ValueError(exception_message)

    @classmethod
    def to_opencv_rotation(cls, device_orientation: 'DeviceOrientation') -> Optional[int]:
        """
        Get the corresponding rotation (cv2 code, e.g. cv2.ROTATE_90_CLOCKWISE) for a device orientation such that when
        applied would take a frame and put it the right way up (StrayScanner datasets).

        :param device_orientation: The device orientation.
        :return: The corresponding rotation code, None if
        """
        if device_orientation == DeviceOrientation.Portrait:
            return cv2.ROTATE_90_CLOCKWISE
        elif device_orientation == DeviceOrientation.LandscapeReverse:
            return cv2.ROTATE_180
        elif device_orientation == DeviceOrientation.PortraitReverse:
            return cv2.ROTATE_90_COUNTERCLOCKWISE
        else:  # device_orientation == DeviceOrientation.Landscape
            return None


class StrayScannerAdaptor(VideoAdaptorBase):
    """Converts a dataset captured with 'Stray Scanner' on an iOS device with a LiDAR sensor to the VTMDataset format."""

    # The files needed for a valid dataset.
    video_filename = 'rgb.mp4'
    camera_matrix_filename = 'camera_matrix.csv'
    camera_trajectory_filename = 'odometry.csv'
    required_files = [video_filename, camera_matrix_filename, camera_trajectory_filename]

    depth_folder = 'depth'
    confidence_map_folder = 'confidence'
    required_folders = [depth_folder, confidence_map_folder]

    depth_confidence_levels = [0, 1, 2]
    valid_depth_map_types = {np.dtype('uint16'), np.dtype('uint32'), np.dtype('uint64'),
                             np.dtype('int32'), np.dtype('int64')}

    def __init__(self, base_path: File, output_path: File, overwrite_ok=False, num_frames=-1, frame_step=1,
                 colmap_options=COLMAPOptions(),
                 resize_to: Optional[Union[int, Size]] = None,
                 depth_confidence_filter_level=0):
        """
        :param base_path: The path to the dataset.
        :param output_path: The path to write the new dataset to.
        :param overwrite_ok: Whether it is okay to overwrite existing adapted dataset.
        :param num_frames: The maximum of frames to process. Set to -1 (default) to process all frames.
        :param frame_step: The frequency to sample frames at for COLMAP and pose optimisation.
            If set to 1, samples all frames (i.e. no effect). Otherwise if set to n > 1, samples every n frames.
        :param colmap_options: The configuration to use for COLMAP if estimating camera parameters.
        :param resize_to: The resolution (height, width) to resize the images to.
            If an int is given, the longest side will be scaled to this value and the shorter side will have its new
            length automatically calculated.
        :param depth_confidence_filter_level: The minimum confidence value (0, 1, or 2) for the corresponding depth
                                              value to be kept. E.g. if set to 1, all pixels in the depth map where the
                                              corresponding pixel in the confidence map is less than 1 will be ignored.
        """
        video_path = pjoin(base_path, StrayScannerAdaptor.video_filename)

        super().__init__(base_path=base_path, output_path=output_path, video_path=video_path, num_frames=num_frames,
                         frame_step=frame_step, overwrite_ok=overwrite_ok, colmap_options=colmap_options,
                         resize_to=resize_to)

        self.depth_confidence_filter_level = depth_confidence_filter_level

        assert depth_confidence_filter_level in self.depth_confidence_levels, \
            f"Confidence filter must be one of the following: {self.depth_confidence_levels}."

        self.camera_trajectory = self._load_camera_trajectory()

        # Note: This step must be done BEFORE the camera trajectory is normalised because the rotation will be reset.
        roll = Rotation.from_quat(self.camera_trajectory.rotations[0]).as_euler('xyz')[-1]
        self.device_orientation = DeviceOrientation.from_angle(roll)

        if self.device_orientation in {DeviceOrientation.Portrait, DeviceOrientation.PortraitReverse}:
            # The above orientations will cause the frames to be rotated 90 degrees, which swaps the frame height and
            # width. So we need to make sure this change is reflected in the target resolution.
            self.target_height, self.target_width = self.target_width, self.target_height

        # TODO: Retain roll? What about pitch? Yaw can be safely discarded so that the scene is facing the
        #  same direction for all scenes. Take angle modulus 90 degrees to account for when device is not held
        #  perfectly level (pretty much always)?
        self.camera_trajectory = self.camera_trajectory.normalise()
        # TODO: Get this working with orientations other than `.Landscape`

    def _load_camera_trajectory(self) -> Trajectory:
        """
        Load the camera poses and.

        :return: The Nx7 camera trajectory.
        """
        # Code adapted from https://github.com/kekeblom/StrayVisualizer/blob/df5f39c750e8eec62b130dc9c8a91bdbcff1d952/stray_visualize.py#L43
        trajectory_path = pjoin(self.base_path, self.camera_trajectory_filename)
        # The first row is the header row, so skip
        trajectory_raw = np.loadtxt(trajectory_path, delimiter=',', skiprows=1)

        trajectory = []

        for line in trajectory_raw:
            # timestamp, frame, ...
            _, _, tx, ty, tz, qx, qy, qz, qw = line
            trajectory.append((qx, qy, qz, qw, tx, ty, tz))

        trajectory = np.asarray(trajectory)
        trajectory = Trajectory(trajectory).inverse()

        return trajectory

    def get_camera_matrix(self) -> np.ndarray:
        scale_x = self.target_width / self.source_width
        scale_y = self.target_height / self.source_height

        intrinsics_path = pjoin(self.base_path, self.camera_matrix_filename)
        camera_matrix = np.loadtxt(intrinsics_path, delimiter=',')

        camera_matrix[0, 0] *= scale_x
        camera_matrix[0, 2] *= scale_x
        camera_matrix[1, 1] *= scale_y
        camera_matrix[1, 2] *= scale_y

        return camera_matrix

    def get_pose(self, index: int) -> np.ndarray:
        return self.camera_trajectory[index]

    def copy_frames(self, output_path: str, num_frames=-1):
        num_frames = self.num_frames if num_frames == -1 else num_frames

        self.extract_video(self.video_path, output_path, num_frames,
                           target_resolution=(self.target_height, self.target_width),
                           rotation=DeviceOrientation.to_opencv_rotation(self.device_orientation))

    def get_depth_map(self, index: int) -> np.ndarray:
        filename = VTMDataset.index_to_filename(index, file_extension='png')
        depth_map_path = pjoin(self.base_path, self.depth_folder, filename)
        depth_map = imageio.v3.imread(depth_map_path)

        if depth_map.dtype not in self.valid_depth_map_types:
            raise RuntimeError(f"Expected depth map of one the following types: {self.valid_depth_map_types}, "
                               f"but got {depth_map.dtype}.")

        confidence_map_path = pjoin(self.base_path, self.confidence_map_folder, filename)
        confidence_map = imageio.v3.imread(confidence_map_path)

        depth_map[confidence_map < self.depth_confidence_filter_level] = 0

        # cv2.resize(...) only works with floating point type arrays.
        original_type = depth_map.dtype
        depth_map = depth_map.astype(np.float32)
        # Creators of StrayScanner suggest that nearest neighbour interpolation should be used.
        depth_map = cv2.resize(depth_map, dsize=(self.target_width, self.target_height),
                               interpolation=cv2.INTER_NEAREST_EXACT)
        depth_map = np.round(depth_map)
        depth_map = depth_map.astype(original_type)

        if rotation := DeviceOrientation.to_opencv_rotation(self.device_orientation):
            depth_map = cv2.rotate(depth_map, rotation)

        return depth_map


def create_folder(*args, exist_ok=False):
    path = pjoin(*args)

    os.makedirs(path, exist_ok=exist_ok)

    return path


def estimate_depth_dpt(rgb_dataset, output_path: str, weights_filename='dpt_hybrid_nyu.pt', optimize=True):
    # TODO: Fix memory leak that happens after using the DPT model.
    #  Memory only gets released after the script exits. I suspect it has something to do with the timm package or
    #  if not that, something in the DPT model code.
    model_path = os.path.join(os.environ['WEIGHTS_PATH'], weights_filename)

    if not os.path.isfile(model_path):
        model_path = os.path.join('weights', weights_filename)

    # TODO: Once memory leak is fixed, re-enable use of cuDNN
    # with cudnn():
    # select device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load network
    # TODO: Make input resolution a parameter.
    net_w = 640
    net_h = 480

    model = dpt.models.DPTDepthModel(
        path=model_path,
        scale=0.000305,
        shift=0.1378,
        invert=True,
        backbone="vitb_rn50_384",
        non_negative=True,
        enable_attention_hooks=False,
    )

    normalization = dpt.transforms.NormalizeImage(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

    transform = transforms.Compose(
        [
            dpt.transforms.Resize(
                net_w,
                net_h,
                resize_target=None,
                keep_aspect_ratio=True,
                ensure_multiple_of=32,
                resize_method="minimal",
                image_interpolation_method=cv2.INTER_CUBIC,
            ),
            normalization,
            dpt.transforms.PrepareForNet(),
        ]
    )

    model.eval()

    if optimize and device == torch.device("cuda"):
        model = model.to(memory_format=torch.channels_last)
        model = model.half()

    model.to(device)

    # create output folder
    os.makedirs(output_path, exist_ok=True)

    for i, image in tqdm(enumerate(rgb_dataset), total=len(rgb_dataset)):
        img = image / 255.0

        img_input = transform({"image": img})["image"]

        # compute
        with torch.no_grad():
            sample = torch.from_numpy(img_input).to(device).unsqueeze(0)

            if optimize and device == torch.device("cuda"):
                sample = sample.to(memory_format=torch.channels_last)
                sample = sample.half()

            prediction = model.forward(sample)
            prediction = (
                torch.nn.functional.interpolate(
                    prediction.unsqueeze(1),
                    size=img.shape[:2],
                    mode="nearest",
                )
                .squeeze()
                .cpu()
                .numpy()
            )

        depth_map = prediction * 1000.0
        depth_map = depth_map.astype(np.uint16)

        imageio.imwrite(os.path.join(output_path, f"{i:06d}.png"), depth_map)


def get_dataset(storage_options: StorageOptions, colmap_options=COLMAPOptions(), pipeline_options=PipelineOptions(),
                output_path: Optional[str] = None, resize_to: Union[int, Size] = 640,
                depth_confidence_filter_level=0) -> VTMDataset:
    """
    Get a VTM formatted dataset or create one from another dataset format.

    :param storage_options: The options that includes the path to the dataset.
    :param colmap_options: The configuration to use for COLMAP if estimating camera pose.
    :param pipeline_options: Configuration including whether to estimate pose or depth, the number of frames to include,
        and the frame step for COLMAP/TSDFFusion.
    :param output_path: (optional) where to save the dataset to if it needs to be created. If `None`, the dataset will
        be created at `storage_options.base_path`.
    :param resize_to: The resolution (height, width) to resize the images to. If an int is given, the longest side will
        be scaled to this value and the shorter side will have its new length automatically calculated.
    :param depth_confidence_filter_level: The minimum confidence value (0, 1, or 2) for the corresponding depth value
     to be kept. E.g. if set to 1, all pixels in the depth map where the corresponding pixel in the confidence map is
     less than 1 will be ignored.
    :return: the VTM formatted dataset.
    """
    dataset_path = storage_options.base_path
    output_path = f"{dataset_path}_vtm" if output_path is None else output_path

    if VTMDataset.is_valid_folder_structure(output_path):
        dataset = VTMDataset(output_path, overwrite_ok=storage_options.overwrite_ok)
    else:
        base_kwargs = dict(
            base_path=dataset_path,
            output_path=output_path,
            num_frames=pipeline_options.num_frames,
            frame_step=pipeline_options.frame_step,
            overwrite_ok=storage_options.overwrite_ok,
            colmap_options=colmap_options
        )

        if TUMAdaptor.is_valid_folder_structure(dataset_path):
            dataset_converter = TUMAdaptor(**base_kwargs)
        elif StrayScannerAdaptor.is_valid_folder_structure(dataset_path):
            dataset_converter = StrayScannerAdaptor(
                **base_kwargs,
                # TODO: Make target image size configurable via cli.
                resize_to=resize_to,
                # TODO: Make depth confidence filter level configurable via cli.
                depth_confidence_filter_level=depth_confidence_filter_level
            )
        elif VideoAdaptor.is_valid_folder_structure(dataset_path):
            path_no_extensions, _ = os.path.splitext(dataset_path)

            dataset_converter = VideoAdaptor(**base_kwargs, resize_to=resize_to)
        elif not os.path.isdir(dataset_path):
            raise RuntimeError(f"Could open the path {dataset_path} or it is not a folder.")
        else:
            raise RuntimeError(f"Could not recognise the dataset format for the dataset at {dataset_path}.")

        dataset = dataset_converter.convert(estimate_pose=pipeline_options.estimate_pose,
                                            estimate_depth=pipeline_options.estimate_depth)

    return dataset
