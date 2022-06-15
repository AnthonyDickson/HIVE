import contextlib
import os
import shutil
import subprocess
import warnings
from os.path import join as pjoin
from pathlib import Path
from typing import Optional, Union, Tuple

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
from video2mesh.geometry import normalise_trajectory
from video2mesh.io import DatasetBase, File, DatasetMetadata, VTMDataset, COLMAPProcessor, ImageFolderDataset, \
    create_masks, Size, VideoMetadata, InvalidDatasetFormatError
from video2mesh.options import COLMAPOptions, StaticMeshOptions
from video2mesh.pose_optimisation import PoseOptimiser, FeatureExtractionOptions
from video2mesh.utils import log, tqdm_imap


class DatasetAdaptor(DatasetBase):
    """Creates a copy of a dataset in the VTMDataset format."""

    def __init__(self, base_path: File, output_path: File, num_frames=-1, frame_step=1, overwrite_ok=False):
        """
        :param base_path: The path to the dataset.
        :param output_path: The path to write the new dataset to.
        :param overwrite_ok: Whether it is okay to overwrite existing adapted dataset.
        :param num_frames: The maximum of frames to process. Set to -1 (default) to process all frames.
        :param frame_step: The frequency to sample frames at for COLMAP and pose optimisation.
            If set to 1, samples all frames (i.e. no effect). Otherwise if set to n > 1, samples every n frames.
        """
        super().__init__(base_path=base_path, overwrite_ok=overwrite_ok)

        self.output_path = output_path
        self.num_frames = num_frames
        self.frame_step = frame_step

    def get_full_num_frames(self) -> int:
        """The number of frames in the non-truncated dataset."""
        raise NotImplementedError

    def get_metadata(self, is_gt: bool) -> DatasetMetadata:
        """
        Get the metadata object for this dataset.

        :param is_gt: Whether the dataset was/will be created using ground truth data for
            the camera parameters and depth maps.
        """
        raise NotImplementedError

    def get_camera_trajectory(self) -> np.ndarray:
        """
        Get the ground truth camera trajectory.
        If the dataset does not have a ground truth camera trajectory, a `NotImplementedError` is raised.

        :return: The (N, 7) ground truth camera trajectory, if it exists.
        """
        trajectory = np.vstack([self.get_pose(i) for i in range(self.num_frames)])
        trajectory = normalise_trajectory(trajectory)

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
        return self._convert(from_gt=True)

    def convert_from_rgb(self, colmap_options=COLMAPOptions()) -> VTMDataset:
        """
        Read a video file and create a VTMDataset with estimated camera parameters and depth maps.

        :return: The newly created dataset object.
        """
        # TODO: Add way to pass arguments to PoseOptimiser object.
        return self._convert(from_gt=False, colmap_options=colmap_options)

    def _convert(self, from_gt: bool, colmap_options=COLMAPOptions()) -> VTMDataset:
        if cached_dataset := self._try_get_cached_dataset(is_gt=from_gt):
            log(f"Found cached dataset at {self.output_path}.")
            return cached_dataset

        output_image_folder, output_depth_folder, output_mask_folder = self._setup_folders()

        log(f"Creating metadata for dataset.")
        metadata = self.get_metadata(is_gt=from_gt)
        metadata_path = pjoin(self.output_path, VTMDataset.metadata_filename)
        metadata.save(metadata_path)

        log(f"Copying RGB frames.")
        self.copy_frames(output_image_folder)

        create_masks(
            DataLoader(ImageFolderDataset(output_image_folder), batch_size=8),
            mask_folder=output_mask_folder,
            overwrite_ok=True
        )

        if from_gt:
            log(f"Copying depth maps.")
            self.copy_depth_maps(output_depth_folder)

            camera_matrix = self.get_camera_matrix()
            camera_trajectory = self.get_camera_trajectory()
        else:
            log(f"Creating depth maps.")
            estimate_depth_dpt(ImageFolderDataset(output_image_folder), output_depth_folder)

            debug_folder = pjoin(self.output_path, 'debug')
            camera_matrix, camera_trajectory = self._estimate_camera_parameters(debug_folder, colmap_options)
            camera_trajectory = self._refine_estimated_camera_parameters(camera_matrix, camera_trajectory,
                                                                         debug_folder, output_image_folder,
                                                                         output_depth_folder, output_mask_folder)

        log(f"Creating camera matrix file.")
        camera_matrix_path = pjoin(self.output_path, VTMDataset.camera_matrix_filename)
        # noinspection PyTypeChecker
        np.savetxt(camera_matrix_path, camera_matrix)

        log(f"Creating camera trajectory file.")
        camera_trajectory_path = pjoin(self.output_path, VTMDataset.camera_trajectory_filename)
        # noinspection PyTypeChecker
        np.savetxt(camera_trajectory_path, camera_trajectory)

        log(f"Created new dataset at {self.output_path}.")

        return VTMDataset(self.output_path, overwrite_ok=self.overwrite_ok)

    def _try_get_cached_dataset(self, is_gt: bool) -> Optional[VTMDataset]:
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

            same_metadata = dataset.metadata == self.get_metadata(is_gt)

            if same_num_frames and same_trajectory_length and same_metadata:
                return dataset

        return None

    def _delete_cache(self):
        if os.path.isdir(self.output_path):
            if self.overwrite_ok:
                warnings.warn(f"Found a dataset in {self.output_path} but it was created with different settings."
                              f"Since `overwrite_ok` is set to `True`, the existing dataset will be deleted.")
                shutil.rmtree(self.output_path)
            else:
                raise RuntimeError(f"The output path {self.output_path} already exists! "
                                   f"Possible fix: Delete or rename the file/folder.")

    def _setup_folders(self) -> Tuple[str, str, str]:
        self._delete_cache()
        os.makedirs(self.output_path, exist_ok=self.overwrite_ok)

        image_folder = create_folder(self.output_path, VTMDataset.rgb_folder, exist_ok=self.overwrite_ok)
        depth_folder = create_folder(self.output_path, VTMDataset.depth_folder, exist_ok=self.overwrite_ok)
        mask_folder = create_folder(self.output_path, VTMDataset.mask_folder, exist_ok=self.overwrite_ok)

        return image_folder, depth_folder, mask_folder

    @staticmethod
    def _get_frame_subset(num_frames, frame_step):
        frames = list(range(num_frames))
        frames_subset = frames[::frame_step]

        if frames_subset[-1] != frames[-1]:
            frames_subset += [frames[-1]]

        return frames, frames_subset

    def _estimate_camera_parameters(self, output_folder: str, colmap_options: COLMAPOptions):
        # TODO: Redirect COLMAP output to log file and print some sort of progress bar.
        colmap_log_file = pjoin(output_folder, 'colmap_logs.txt')

        if os.path.exists(output_folder):
            shutil.rmtree(output_folder)

        colmap_folder = create_folder(output_folder, 'colmap')
        colmap_rgb_path = create_folder(colmap_folder, 'rgb')
        colmap_workspace_path = create_folder(colmap_folder, 'workspace')

        frames, frames_subset = self._get_frame_subset(self.get_full_num_frames(), self.frame_step)

        log("Copying RGB frames for COLMAP...")

        self.copy_frames(colmap_rgb_path, self.get_full_num_frames())

        if self.frame_step > 1:
            for index in set(frames).difference(frames_subset):
                os.remove(pjoin(colmap_rgb_path, VTMDataset.index_to_filename(index)))

            for dst_index, src_index in enumerate(frames_subset):
                src_path = pjoin(colmap_rgb_path, VTMDataset.index_to_filename(src_index))
                dst_path = pjoin(colmap_rgb_path, VTMDataset.index_to_filename(dst_index))

                shutil.move(src_path, dst_path)

        log(f"Running COLMAP... This might take a while!")
        log(f"Check {colmap_log_file} for the logs.")

        colmap_processor = COLMAPProcessor(image_path=colmap_rgb_path, workspace_path=colmap_workspace_path,
                                           colmap_options=colmap_options)

        with open(colmap_log_file, 'w') as f, contextlib.redirect_stdout(f), contextlib.redirect_stderr(f):
            colmap_processor.run()

        camera_matrix, camera_trajectory = colmap_processor.load_camera_params()

        return camera_matrix, camera_trajectory

    def _refine_estimated_camera_parameters(self, camera_matrix: np.ndarray, camera_trajectory: np.ndarray,
                                            output_folder: str, image_folder: str, depth_folder: str, mask_folder: str):
        def copy_frame_set(src_path, dst_path):
            def copy(dst_src_indices):
                dst_index, src_index = dst_src_indices

                src = dataset.image_paths[src_index]
                dst = pjoin(dst_path, VTMDataset.index_to_filename(dst_index))
                shutil.copy(src, dst)

            dataset = ImageFolderDataset(src_path)
            tqdm_imap(copy, list(enumerate(frames_subset)))

        frames, frames_subset = self._get_frame_subset(self.num_frames, self.frame_step)

        tmp_dataset_path = create_folder(output_folder, 'pose_optim_vtm')

        log("Copying RGB frames for pose optimisation...")
        tmp_rgb_path = create_folder(tmp_dataset_path, VTMDataset.rgb_folder)
        copy_frame_set(image_folder, tmp_rgb_path)

        log("Copying depth maps for pose optimisation...")
        tmp_depth_path = create_folder(tmp_dataset_path, VTMDataset.depth_folder)
        copy_frame_set(depth_folder, tmp_depth_path)

        log("Copying masks for pose optimisation...")
        tmp_mask_path = create_folder(tmp_dataset_path, VTMDataset.mask_folder)
        copy_frame_set(mask_folder, tmp_mask_path)

        temp_metadata = self.get_metadata(is_gt=False)
        temp_metadata.num_frames = len(frames_subset)
        temp_metadata.save(pjoin(tmp_dataset_path, VTMDataset.metadata_filename))

        # noinspection PyTypeChecker
        np.savetxt(pjoin(tmp_dataset_path, VTMDataset.camera_matrix_filename), camera_matrix)
        # noinspection PyTypeChecker
        np.savetxt(pjoin(tmp_dataset_path, VTMDataset.camera_trajectory_filename), camera_trajectory)

        temp_dataset = VTMDataset(tmp_dataset_path, overwrite_ok=True)

        # TODO: Allow optimisation options to be set via CLI?
        optimiser = PoseOptimiser(
            temp_dataset,
            feature_extraction_options=FeatureExtractionOptions(min_features=40),
            debug=True
        )
        optimised_trajectory, _, _ = optimiser.run(num_frames=temp_dataset.num_frames)

        interpolated_poses = np.zeros((self.num_frames, 7))
        interpolated_poses[:, 3] = 1.0  # Ensure all quaternions are initialised as unit quaternions.
        interpolated_poses[frames_subset] = optimised_trajectory

        for start, end in zip(frames_subset[:-1], frames_subset[1:]):
            start_rotation = interpolated_poses[start, :4]
            end_rotation = interpolated_poses[end, :4]
            start_position = interpolated_poses[start, 4:]
            end_position = interpolated_poses[end, 4:]

            key_frame_times = [0, 1]
            times_to_interpolate = np.linspace(0, 1, num=end - start + 1)

            slerp = Slerp(times=key_frame_times, rotations=Rotation.from_quat([start_rotation, end_rotation]))
            lerp = interp1d(key_frame_times, [start_position, end_position], axis=0)

            interpolated_poses[start:end + 1, 4:] = lerp(times_to_interpolate)
            interpolated_poses[start:end + 1, :4] = slerp(times_to_interpolate).as_quat()

        return interpolated_poses


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

    def __init__(self, base_path: File, output_path: File, num_frames=-1, frame_step=1,
                 is_16_bit=True, overwrite_ok=False):
        """
        :param base_path: The path to folder containing the dataset.
        :param output_path: The path to write the new dataset to.
        :param is_16_bit: Whether the images are stored with 16-bit values or 32-bit values.
        :param overwrite_ok: Whether it is okay to overwrite existing depth maps and/or instance segmentation masks.
        """
        super().__init__(base_path=base_path, output_path=output_path,
                         num_frames=num_frames, frame_step=frame_step,
                         overwrite_ok=overwrite_ok)

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

        self.camera_trajectory = normalise_trajectory(self.camera_trajectory)
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

        return image_filenames_subset, depth_map_subset, trajectory_subset

    def get_frame_path(self, index) -> str:
        return pjoin(self.base_path, self.rgb_folder, self.image_filenames[index])

    def get_depth_map_path(self, index) -> str:
        return pjoin(self.base_path, self.depth_folder, self.depth_filenames[index])

    def get_full_num_frames(self):
        a, _, _ = self._get_synced_frame_data()
        return len(a)

    def get_metadata(self, is_gt: bool):
        # This gets the default value for `depth_mask_dilation_iterations`.
        depth_mask_dilation_iterations = StaticMeshOptions().depth_mask_dilation_iterations

        return DatasetMetadata(num_frames=self.num_frames, frame_step=self.frame_step,
                               fps=self.fps, width=self.width, height=self.height, is_gt=is_gt,
                               depth_mask_dilation_iterations=depth_mask_dilation_iterations,
                               depth_scale=VTMDataset.depth_scaling_factor)

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


class VideoAdaptorBase(DatasetAdaptor):
    """Base class for adaptors of video datasets."""

    def __init__(self, base_path: Union[str, Path], output_path: Union[str, Path], video_path: Union[str, Path],
                 num_frames=-1, frame_step=1,
                 overwrite_ok=False,
                 resize_to: Optional[Union[int, Size]] = None):
        """
        :param base_path: The path to the video file.
        :param output_path: The path to write the new dataset to.
        :param video_path: The path to the RGB video.
        :param overwrite_ok: Whether it is okay to overwrite existing depth maps and/or instance segmentation masks.
        :param resize_to: The resolution (width, height) to resize the images to.
            If an int is given, the longest side will be scaled to this value and the shorter side will have its new
            length automatically calculated.
        """
        super().__init__(base_path=base_path, output_path=output_path,
                         num_frames=num_frames, frame_step=frame_step, overwrite_ok=overwrite_ok)

        self.video_path = video_path

        with open_video(self.video_path) as video:
            self.num_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT)) if self.num_frames == -1 else self.num_frames
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
            log(f"Will resize frames from {self.source_width}x{self.source_height} to "
                f"{self.target_width}x{self.target_height} (width, height).")

    def get_metadata(self, is_gt: bool) -> DatasetMetadata:
        with open_video(self.video_path) as video:
            fps = float(video.get(cv2.CAP_PROP_FPS))

        height, width = self.target_height, self.target_width

        video_metadata = VideoMetadata(self.video_path, width=width, height=height, num_frames=self.num_frames, fps=fps)

        # This gets the default value for `depth_mask_dilation_iterations`.
        depth_mask_dilation_iterations = StaticMeshOptions().depth_mask_dilation_iterations

        return DatasetMetadata(num_frames=video_metadata.num_frames, fps=video_metadata.fps, width=width, height=height,
                               is_gt=is_gt, frame_step=self.frame_step,
                               depth_mask_dilation_iterations=depth_mask_dilation_iterations,
                               depth_scale=VTMDataset.depth_scaling_factor)

    def get_full_num_frames(self):
        with open_video(self.video_path) as video:
            return int(video.get(cv2.CAP_PROP_FRAME_COUNT))

    def get_frame(self, index: int) -> np.ndarray:
        with open_video(self.video_path) as video:
            video.set(cv2.CAP_PROP_POS_FRAMES, index - 1)

            read_frame, frame = video.read()

        if read_frame:
            return cv2.cvtColor(cv2.resize(frame, (self.target_width, self.target_height)), cv2.COLOR_BGR2RGB)
        else:
            raise RuntimeError(f"Could not read frame {index} (zero-based index) from the video {self.video_path}.")

    def copy_frames(self, output_path: str, num_frames=-1):
        num_frames = self.num_frames if num_frames == -1 else num_frames

        extract_video(self.video_path, output_path, num_frames)


class VideoAdaptor(VideoAdaptorBase):
    """Converts an RGB video to the VTMDataset format."""

    def __init__(self, base_path: Union[str, Path], output_path: Union[str, Path],
                 num_frames=-1, frame_step=1,
                 overwrite_ok=False,
                 resize_to: Optional[Union[int, Size]] = None):
        """
        :param base_path: The path to the video file.
        :param output_path: The path to write the new dataset to.
        :param overwrite_ok: Whether it is okay to overwrite existing depth maps and/or instance segmentation masks.
        :param resize_to: The resolution (width, height) to resize the images to.
            If an int is given, the longest side will be scaled to this value and the shorter side will have its new
            length automatically calculated.
        """
        base_path = Path(base_path).parent
        video_filename = Path(base_path).name
        video_path = pjoin(base_path, video_filename)

        super().__init__(base_path=base_path, output_path=output_path, video_path=video_path, resize_to=resize_to,
                         num_frames=num_frames, frame_step=frame_step, overwrite_ok=overwrite_ok)

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
            files = list(filter(os.path.isfile, items))

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

    def __init__(self, base_path: Union[str, Path], output_path: Union[str, Path],
                 num_frames=-1, frame_step=1,
                 overwrite_ok=False,
                 resize_to: Optional[Union[int, Size]] = None,
                 depth_confidence_filter_level=0):
        """
        :param base_path: The path to the dataset.
        :param output_path: The path to write the new dataset to.
        :param overwrite_ok: Whether it is okay to overwrite existing depth maps and/or instance segmentation masks.
        :param resize_to: The resolution (width, height) to resize the images to.
            If an int is given, the longest side will be scaled to this value and the shorter side will have its new
            length automatically calculated.
        :param depth_confidence_filter_level: The minimum confidence value (0, 1, or 2) for the corresponding depth
                                              value to be kept. E.g. if set to 1, all pixels in the depth map where the
                                              corresponding pixel in the confidence map is less than 1 will be ignored.
        """
        video_path = pjoin(base_path, StrayScannerAdaptor.video_filename)

        super().__init__(base_path=base_path, output_path=output_path, video_path=video_path, resize_to=resize_to,
                         num_frames=num_frames, frame_step=frame_step, overwrite_ok=overwrite_ok)

        self.depth_confidence_filter_level = depth_confidence_filter_level

        assert depth_confidence_filter_level in self.depth_confidence_levels, \
            f"Confidence filter must be one of the following: {self.depth_confidence_levels}."

        self.camera_trajectory = self._load_camera_trajectory()

    def _load_camera_trajectory(self):
        """
        Load the camera poses.

        :return: The Nx6 matrix where each row contains the rotation in axis-angle format and the translation vector.
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
        trajectory = normalise_trajectory(trajectory)

        # TODO: Account for device orientation. The iPhone could have been help in portrait or landscape...

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

    def get_depth_map(self, index: int) -> np.ndarray:
        filename = f"{index:06d}"
        depth_map_path = pjoin(self.base_path, self.depth_folder, f"{filename}.npy")
        depth_map = np.load(depth_map_path)

        if depth_map.dtype != np.uint16:
            raise RuntimeError(f"Expected 16-bit depth maps, got {depth_map.dtype}.")

        confidence_map_path = pjoin(self.base_path, self.confidence_map_folder, f"{filename}.png")
        confidence_map = imageio.v3.imread(confidence_map_path)

        depth_map[confidence_map < self.depth_confidence_filter_level] = 0

        depth_map = cv2.resize(depth_map, (self.target_width, self.target_height))

        return depth_map


def create_folder(*args, exist_ok=False):
    path = pjoin(*args)

    os.makedirs(path, exist_ok=exist_ok)

    return path


def extract_video(path_to_video: str, output_path: str, num_frames: int = -1):
    ffmpeg_command = ['ffmpeg', '-i', path_to_video]

    if num_frames != -1:
        ffmpeg_command += ['-frames:v', str(num_frames)]

    ffmpeg_command += ['-start_number', str(0), pjoin(output_path, '%06d.png')]

    process = subprocess.Popen(ffmpeg_command)

    if (return_code := process.wait()) != 0:
        raise RuntimeError(f"Something went wrong with ffmpeg (return code {return_code}), "
                           f"check the logs for more details.")


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
                    mode="bicubic",
                    align_corners=False,
                )
                    .squeeze()
                    .cpu()
                    .numpy()
            )

        depth_map = prediction * 1000.0
        depth_map = depth_map.astype(np.uint16)

        imageio.imwrite(os.path.join(output_path, f"{i:06d}.png"), depth_map)


def calculate_target_resolution(source_hw, target_hw):
    """
    Calculate the target resolution and perform some sanity checks.

    :param source_hw: The resolution of the input frames. These are used if the target resolution is given as a
        single value indicating the desired length of the longest side of the images.
    :param target_hw: The resolution (width, height) to resize the images to.
    :return: The target resolution as a 2-tuple (height, width).
    """
    if isinstance(target_hw, int):
        # Cast results to int to avoid warning highlights in IDE.
        longest_side = int(np.argmax(source_hw))
        shortest_side = int(np.argmin(source_hw))

        new_size = [0, 0]
        new_size[longest_side] = target_hw

        scale_factor = target_hw / source_hw[longest_side]
        new_size[shortest_side] = int(source_hw[shortest_side] * scale_factor)

        target_hw = new_size
    elif isinstance(target_hw, tuple):
        if len(target_hw) != 2:
            raise ValueError(f"The target resolution must be a 2-tuple, but got a {len(target_hw)}-tuple.")

        if not isinstance(target_hw[0], int) or not isinstance(target_hw[1], int):
            raise ValueError(f"Expected target resolution to be a 2-tuple of integers, but got a tuple of"
                             f" ({type(target_hw[0])}, {type(target_hw[1])}).")

    target_orientation = 'portrait' if np.argmax(target_hw) == 0 else 'landscape'
    source_orientation = 'portrait' if np.argmax(source_hw) == 0 else 'landscape'

    if target_orientation != source_orientation:
        warnings.warn(
            f"The input images appear to be in {source_orientation} ({source_hw[1]}x{source_hw[0]}), "
            f"but they are being resized to what appears to be "
            f"{target_orientation} ({target_hw[1]}x{target_hw[0]})")

    source_aspect = source_hw[1] / source_hw[0]
    target_aspect = target_hw[1] / target_hw[0]

    if not np.isclose(source_aspect, target_aspect):
        warnings.warn(f"The aspect ratio of the source video is {source_aspect.as_integer_ratio()}, "
                      f"however the aspect ratio of the target resolution is {target_aspect}. "
                      f"This may lead to stretching in the images.")

    return target_hw


@contextlib.contextmanager
def open_video(video_path):
    video = cv2.VideoCapture(video_path)

    try:
        yield video
    finally:
        if video.isOpened():
            video.release()