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

import abc
import contextlib
import cv2
import datetime
import imageio
import json
import logging
import numpy as np
import os
import struct
import subprocess
import torch
from PIL import Image
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.engine import DefaultPredictor
from os.path import join as pjoin
from pathlib import Path
from scipy.spatial.transform import Rotation
from torch.utils.data import DataLoader as TorchDataLoader, Dataset as TorchDataset
from tqdm import tqdm
from typing import Union, Tuple, Optional, Callable, IO, List

from hive.geometric import Trajectory, get_pose_components, world2image, pose_vec2mat, point_cloud_from_depth
from hive.image_processing import dilate_mask, calculate_target_resolution
from hive.options import COLMAPOptions, MaskDilationOptions
from hive.types import File
from hive.utils import tqdm_imap, check_domain, Domain
from third_party.colmap.scripts.python.read_dense import read_array as load_colmap_depth_map
from third_party.colmap.scripts.python.read_write_model import Image as COLMAPImage
from third_party.colmap.scripts.python.read_write_model import read_model


def load_raw_float32_image(file_name):
    """
    Load image from binary file in the same way as read in C++ with
    #include "compphotolib/core/CvUtil.h"
    freadimg(fileName, image);

    :param file_name: Where to load the image from.
    :return: The image.
    """
    with open(file_name, "rb") as f:
        CV_CN_MAX = 512
        CV_CN_SHIFT = 3
        CV_32F = 5
        I_BYTES = 4
        Q_BYTES = 8

        h = struct.unpack("i", f.read(I_BYTES))[0]
        w = struct.unpack("i", f.read(I_BYTES))[0]

        cv_type = struct.unpack("i", f.read(I_BYTES))[0]
        pixel_size = struct.unpack("Q", f.read(Q_BYTES))[0]
        d = ((cv_type - CV_32F) >> CV_CN_SHIFT) + 1
        assert d >= 1
        d_from_pixel_size = pixel_size // 4
        if d != d_from_pixel_size:
            raise Exception(
                "Incompatible pixel_size(%d) and cv_type(%d)" % (pixel_size, cv_type)
            )
        if d > CV_CN_MAX:
            raise Exception("Cannot save image with more than 512 channels")

        data = np.frombuffer(f.read(), dtype=np.float32)
        result = data.reshape(h, w) if d == 1 else data.reshape(h, w, d)
        return result


def save_raw_float32_image(file_name, image):
    """
    Save image to binary file, so that it can be read in C++ with
        #include "compphotolib/core/CvUtil.h"
        freadimg(fileName, image);

    :param file_name: Where to save the image to.
    :param image: The image to save.
    """
    with open(file_name, "wb") as f:
        CV_CN_MAX = 512
        CV_CN_SHIFT = 3
        CV_32F = 5

        dims = image.shape

        d = 1
        if len(dims) == 2:
            h, w = image.shape
            float32_image = np.transpose(image).astype(np.float32)
        else:
            h, w, d = image.shape
            float32_image = np.transpose(image, [2, 1, 0]).astype("float32")

        cv_type = CV_32F + ((d - 1) << CV_CN_SHIFT)

        pixel_size = d * 4

        if d > CV_CN_MAX:
            raise Exception("Cannot save image with more than 512 channels")
        f.write(struct.pack("i", h))
        f.write(struct.pack("i", w))
        f.write(struct.pack("i", cv_type))
        f.write(struct.pack("Q", pixel_size))  # Write size_t ~ uint64_t

        # Set buffer size to 16 MiB to hide the Python loop overhead.
        buffer_size = max(16 * 1024 ** 2 // image.itemsize, 1)

        for chunk in np.nditer(
                float32_image,
                flags=["external_loop", "buffered", "zerosize_ok"],
                buffersize=buffer_size,
                order="F",
        ):
            f.write(chunk.tobytes("C"))


class BatchPredictor(DefaultPredictor):
    """Run d2 on a list of images."""

    def __call__(self, images):
        """Run d2 on a list of images.

        Args:
            images (list): BGR images of the expected shape: 720x1280
        """
        with torch.no_grad():  # https://github.com/sphinx-doc/sphinx/issues/4258
            inputs = []

            for image in images:
                # Apply pre-processing to image.
                if self.input_format == "RGB":
                    # whether the model expects BGR inputs or RGB
                    image = image[:, :, ::-1]
                height, width = image.shape[:2]
                image = self.aug.get_transform(image).apply_image(image)
                image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))

                inputs.append({"image": image, "height": height, "width": width})

            predictions = self.model(inputs)

        return predictions


def create_masks(rgb_loader: TorchDataLoader, mask_folder: Union[str, Path], for_colmap=False,
                 filename_fmt: Optional[Callable[[int], str]] = None):
    """
    Create instance segmentation masks for the given RGB video sequence and save the masks to disk.

    :param rgb_loader: The PyTorch DataLoader that loads the RGB frames (no data augmentations applied).
    :param mask_folder: The path to save the masks to.
    :param for_colmap: Whether the masks are intended for use with COLMAP or 3D video generation.
        Masks will be black and white with the background coloured white and using the
        corresponding input image's filename.
    :param filename_fmt: (optional) a function that generates a frame filename from the frame index,
        e.g. 123 -> '000123.png'.
    """
    logging.info(f"Creating masks...")

    cfg = get_cfg()

    if not torch.cuda.is_available():
        cfg.MODEL.DEVICE = 'cpu'

    # add project-specific config (e.g., TensorMask) here if you're not running a model in detectron2's core library
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
    # Find a model from detectron2's model zoo. You can use the https://dl.fbaipublicfiles... url as well
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
    dataset_metadata = MetadataCatalog.get(cfg.DATASETS.TRAIN[0])
    class_names = dataset_metadata.thing_classes
    logging.debug(f"Instance Segmentation Model Classes: {class_names}")

    person_label = class_names.index('person')
    predictor = BatchPredictor(cfg)

    logging.info(f"Creating segmentation masks...")
    i = 0

    # noinspection PyTypeChecker
    with tqdm(total=len(rgb_loader.dataset)) as progress_bar:
        for image_batch in rgb_loader:
            outputs = predictor(image_batch.numpy())

            for output in outputs:
                matching_masks = output['instances'].get('pred_classes') == person_label
                people_masks = output['instances'].get('pred_masks')[matching_masks]

                if for_colmap:
                    combined_masks = 255 * np.ones_like(image_batch[0].numpy(), dtype=np.uint8)
                    combined_masks = combined_masks[:, :, 0]

                    for mask in people_masks.cpu().numpy():
                        combined_masks[mask] = 0
                else:
                    combined_masks = np.zeros_like(image_batch[0].numpy(), dtype=np.uint8)
                    combined_masks = combined_masks[:, :, 0]

                    for j, mask in enumerate(people_masks.cpu().numpy()):
                        combined_masks[mask] = j + 1

                if filename_fmt:
                    output_filename = filename_fmt(i)
                elif for_colmap:
                    output_filename = f"{rgb_loader.dataset.image_filenames[i]}.png"
                else:
                    output_filename = f"{i:06d}.png"

                Image.fromarray(combined_masks).convert('L').save(pjoin(mask_folder, output_filename))

                i += 1
                progress_bar.update()


class COLMAPProcessor:
    """
    Estimates camera trajectory and intrinsic parameters via COLMAP.
    """

    def __init__(self, image_path: File, workspace_path: File, colmap_options=COLMAPOptions(),
                 colmap_mask_folder='masks'):
        self.image_path = image_path
        self.workspace_path = workspace_path
        self.colmap_options = colmap_options
        self.mask_folder = colmap_mask_folder

    @property
    def mask_path(self):
        return pjoin(self.workspace_path, self.mask_folder)

    @property
    def sparse_path(self) -> str:
        """The path to the sparse reconstruction."""
        return pjoin(self.workspace_path, 'sparse')

    @property
    def dense_path(self) -> str:
        """The path to the dense reconstruction."""
        return pjoin(self.workspace_path, 'dense')

    @property
    def probably_has_results(self) -> bool:
        recon_result_path = pjoin(self.sparse_path, '0')
        min_files_for_recon = 4

        return os.path.isdir(self.sparse_path) and len(os.listdir(self.sparse_path)) > 0 and \
            (os.path.isdir(recon_result_path) and len(os.listdir(recon_result_path)) >= min_files_for_recon)

    def run(self):
        os.makedirs(self.workspace_path, exist_ok=True)
        os.makedirs(self.mask_path, exist_ok=True)

        if len(os.listdir(self.mask_path)) == 0:
            logging.info(f"Could not find masks in folder: {self.mask_path}.")
            logging.info(f"Creating masks for COLMAP...")
            rgb_loader = TorchDataLoader(ImageFolderDataset(self.image_path), batch_size=8, shuffle=False)
            create_masks(rgb_loader, self.mask_path, for_colmap=True)
        else:
            logging.info(f"Found {len(os.listdir(self.mask_path))} masks in {self.mask_path}.")

        logging.info("Running COLMAP for real this time, this may take a while...")

        command = self.get_command()

        # TODO: Check that COLMAP is using GPU
        with subprocess.Popen(command, stdout=subprocess.PIPE, bufsize=1, universal_newlines=True) as p:
            for line in p.stdout:
                logging.debug(line.rstrip('\n'))

        if (return_code := p.wait()) != 0:
            raise RuntimeError(f"COLMAP exited with code {return_code}.")

    def get_command(self, return_as_string=False):
        """
        Build the command for running COLMAP .
        Also validates the paths in the options and raises an exception if any of the specified paths are invalid.

        :param return_as_string: Whether to return the command as a single string, or as an array.
        :return: The COLMAP command.
        """
        options = self.colmap_options

        assert os.path.isfile(options.binary_path), f"Could not find COLMAP binary at location: {options.binary_path}."
        assert os.path.isdir(self.workspace_path), f"Could open workspace path: {self.workspace_path}."
        assert os.path.isdir(self.image_path), f"Could open image folder: {self.image_path}."

        command = [options.binary_path, 'automatic_reconstructor',
                   '--workspace_path', self.workspace_path,
                   '--image_path', self.image_path,
                   '--vocab_tree_path', self.colmap_options.vocab_path,
                   '--single_camera', 1 if options.is_single_camera else 0,  # COLMAP expects 1 for True, 0 for False.
                   '--dense', 1 if options.dense else 0,
                   '--quality', options.quality]

        if self.mask_path is not None:
            assert os.path.isdir(self.mask_path), f"Could not open mask folder: {self.mask_path}."
            command += ['--mask_path', self.mask_path]

        command = list(map(str, command))

        return ' '.join(command) if return_as_string else command

    def _load_model(self):
        models = sorted(os.listdir(self.sparse_path))
        num_models = len(models)

        if num_models == 1:
            sparse_recon_path = pjoin(self.sparse_path, models[0])
        else:
            raise RuntimeError(
                f"COLMAP reconstructed {num_models} models when 1 was expected meaning that the camera trajectory could not be estimated for the entire video."
                f"This may be due to COLMAP using a bad random initial guess of the camera parameters and sometimes can be fixed by running the program again. "
                f"Another potential fix is to try increase the quality setting, e.g. add `--quality medium` to your command in the terminal. "
                f"Otherwise, it is likely due to the video not having the camera movement that COLMAP needs.")

        logging.info(f"Reading COLMAP model from {sparse_recon_path}...")
        cameras, images, points3d = read_model(sparse_recon_path, ext=".bin")

        return cameras, images, points3d

    def load_camera_params(self, raw_pose: bool = True) -> Tuple[np.ndarray, Trajectory]:
        """
        Load the camera intrinsic and extrinsic parameters from a COLMAP sparse reconstruction model.
        :param raw_pose: Whether to use the raw pose data straight from COLMAP.
        :return: A 2-tuple containing the camera matrix (intrinsic parameters) and camera trajectory (extrinsic
            parameters). Each row in the camera trajectory consists of the rotation as a quaternion and the
            translation vector, in that order.
        """
        cameras, images, points3d = self._load_model()

        f, cx, cy, _ = cameras[1].params  # cameras is a dict, COLMAP indices start from one.

        intrinsic = np.eye(3)
        intrinsic[0, 0] = f
        intrinsic[1, 1] = f
        intrinsic[0, 2] = cx
        intrinsic[1, 2] = cy
        logging.info("Read intrinsic parameters.")

        extrinsic = dict()

        if raw_pose:
            for image in images.values():
                # COLMAP quaternions seem to be stored in scalar first format.
                # However, rather than assuming the format we can just rely on the provided function to convert to a
                # rotation matrix, and use SciPy to convert that to a quaternion in scalar last format.
                # This avoids any future issues if the quaternion format ever changes.
                r = Rotation.from_matrix(image.qvec2rotmat()).as_quat()
                t = image.tvec

                extrinsic[self._get_index_from_image(image)] = np.hstack((r, t))
        else:
            # Code adapted from https://github.com/facebookresearch/consistent_depth
            # According to some comments in the above code, "Note that colmap uses a different coordinate system
            # where y points down and z points to the world." The below rotation apparently puts the poses back into
            # a 'normal' coordinate frame.
            colmap_to_normal = np.diag([1, -1, 1])  # I think TUM + TSDFFusion work in X right, Y up and Z forward

            for image in images.values():
                R = image.qvec2rotmat()
                t = image.tvec.reshape(-1, 1)

                R, t = R.T, -R.T.dot(t)
                R = colmap_to_normal.dot(R).dot(colmap_to_normal.T)
                t = colmap_to_normal.dot(t)
                t = t.squeeze()

                r = Rotation.from_matrix(R).as_quat()

                extrinsic[self._get_index_from_image(image)] = np.hstack((r, t))

        frame_count = self._get_frame_count()
        pose_count = len(extrinsic)

        if pose_count < frame_count:
            logging.info(f"COLMAP only estimated pose data for {pose_count} frames out of {frame_count}, "
                         f"interpolating missing pose data...")
            extrinsic = Trajectory.create_by_interpolating(extrinsic, frame_count=frame_count)
        else:
            extrinsic = np.asarray([extrinsic[index] for index in sorted(extrinsic.keys())])
            extrinsic = Trajectory(extrinsic)

        logging.info(f"Read extrinsic parameters for {len(extrinsic)} frames.")

        return intrinsic, extrinsic

    def _get_frame_count(self) -> int:
        files = sorted(os.listdir(self.image_path))

        indices = [self._get_index_from_filename(filename) for filename in files]

        return max(indices) + 1

    def _get_index_from_filename(self, filename: str) -> int:
        filename_without_extension = filename.split('.')[0]

        return int(filename_without_extension)

    def _get_index_from_image(self, image: COLMAPImage) -> int:
        return self._get_index_from_filename(filename=image.name)

    def get_sparse_depth_maps(self, camera_matrix: np.ndarray, camera_poses: Trajectory) -> np.ndarray:
        """
        Recover sparse depth maps from the COLMAP reconstruction.

        :param camera_matrix: 3x3 camera intrinsics matrix.
        :param camera_poses: Nx7 camera poses matrix (quaterion, position vector).
        :return: The NxHxW tensor containing the HxW depth maps.
        """
        cameras, images, points3d = self._load_model()
        K = camera_matrix.copy()
        camera_poses_homogeneous = camera_poses.to_homogenous_transforms()

        first_image_id = next(iter(images))
        first_image = images[first_image_id]
        source_image_shape = cv2.imread(pjoin(self.image_path, first_image.name)).shape[:2]

        depth_maps = np.zeros((len(camera_poses), *source_image_shape), dtype=np.float32)

        for image_data in tqdm(images.values()):
            points = [points3d[point3d_id].xyz for point3d_id in image_data.point3D_ids if point3d_id != -1]
            points = np.asarray(points)

            index = self._get_index_from_image(image_data)
            pose = camera_poses_homogeneous[index]
            R, t = get_pose_components(pose)
            projected_points, depth = world2image(points, K, R, t)

            valid_points = (projected_points[:, 0] > 0) & (projected_points[:, 0] < source_image_shape[1]) & \
                           (projected_points[:, 1] > 0) & (projected_points[:, 1] < source_image_shape[0])

            if valid_points.sum() < 1:
                logging.debug(f"COLMAP image data for frame {image_data.name} has no valid points, skipping...")
                continue

            valid_projected_points = projected_points[valid_points]
            valid_depth = depth[valid_points]

            u, v = valid_projected_points.T

            depth_maps[index, v, u] = valid_depth

        return depth_maps

    def get_dense_depth_maps(self, resize_to: Union[int, Tuple[int, int]] = None) -> np.ndarray:
        """
        Get the depth maps from the dense reconstruction.

        :param resize_to: The resolution (height, width) to resize the depth maps to.
            If an int is given, the longest side will be scaled to this value and the shorter side will have its new
            length automatically calculated.
        :return:
        """
        path_to_depth_maps = pjoin(self.dense_path, '0', 'stereo', 'depth_maps')

        if not os.path.isdir(path_to_depth_maps):
            raise NotADirectoryError(f"Could not find or open a folder at {path_to_depth_maps}. "
                                     f"Did you run COLMAP with `dense = True`?")

        if len(os.listdir(path_to_depth_maps)) == 0:
            raise FileNotFoundError(f"Did not find any depth maps in the folder {path_to_depth_maps}. "
                                    f"Did you run COLMAP with `dense = True`?")

        depth_map_filenames = sorted(os.listdir(path_to_depth_maps))

        if resize_to is not None:
            source_height, source_width = load_colmap_depth_map(pjoin(path_to_depth_maps, depth_map_filenames[0])).shape
            target_height, target_width = calculate_target_resolution((source_height, source_width), resize_to)

            def load_depth_map(filename: str) -> np.ndarray:
                path = pjoin(path_to_depth_maps, filename)
                depth_map = load_colmap_depth_map(path)
                depth_map = cv2.resize(depth_map, (target_width, target_height), interpolation=cv2.INTER_NEAREST_EXACT)

                return depth_map
        else:
            def load_depth_map(filename: str) -> np.ndarray:
                path = pjoin(path_to_depth_maps, filename)
                depth_map = load_colmap_depth_map(path)

                return depth_map

        depth_maps = tqdm_imap(load_depth_map, depth_map_filenames)
        depth_maps = np.asarray(depth_maps)

        max_depth = np.quantile(depth_maps, 0.95)
        depth_maps[depth_maps < 0] = 0
        depth_maps[depth_maps > max_depth] = 0

        return depth_maps


class ImageFolderDataset(TorchDataset):
    def __init__(self, base_dir, transform=None):
        """
        :param base_dir: The path to the folder containing images.
        :param transform: (optional) a transform to apply each image. The transform must accept a numpy array as input.
        """
        assert os.path.isdir(base_dir), f"Could not find the folder: {base_dir}"

        self.base_dir = base_dir
        self.transform = transform

        filenames = list(sorted(os.listdir(base_dir)))
        assert len(filenames) > 0, f"No files found in the folder: {base_dir}"

        self.image_filenames = filenames
        self.image_paths = [pjoin(base_dir, filename) for filename in filenames]

    def __getitem__(self, idx) -> np.ndarray:
        image_path = self.image_paths[idx]

        if image_path.endswith('.raw'):
            image = load_raw_float32_image(image_path)
        else:
            image = Image.open(image_path)

            if image.mode == 'I':
                image = image.convert('I;16')
            elif image.mode != 'L' and image.mode != 'I;16':
                image = image.convert('RGB')

            # noinspection PyTypeChecker
            image = np.asarray(image)

        if self.transform:
            image = self.transform(image)

        return image

    def __len__(self):
        return len(self.image_paths)


class VideoMetadata:
    """Information about a video file."""

    def __init__(self, path: File, width: int, height: int, num_frames: int, fps: float):
        """
        :param path: The path to the video.
        :param width: The width of the video frames.
        :param height: The height of the video frames.
        :param num_frames: The number of frames in the video sequence.
        :param fps: The frame rate of the video.
        """
        self.path = path
        self.width = width
        self.height = height
        self.num_frames = num_frames
        self.fps = fps

    @property
    def length_seconds(self):
        """
        The length of the video in seconds.
        """
        return self.num_frames / self.fps

    @property
    def duration(self):
        """
        The length of the video as a datetime.timedelta object.
        """
        return datetime.timedelta(seconds=self.length_seconds)

    def __repr__(self):
        return f"{self.__class__.__name__}(path={self.path}, width={self.width}, height={self.height}, num_frames={self.num_frames}, fps={self.fps})"

    def __str__(self):
        return f"Video at {self.path}: {self.num_frames:.0f} frames, " \
               f"{self.width:.0f} x {self.height:.0f} @ {self.fps} fps with duration of {self.duration}."

    def save(self, f: Union[File, IO]):
        """
        Write the metadata to disk as a JSON file.

        :param f: The file pointer or path to the write to.
        """
        if isinstance(f, (str, Path)):
            with open(f) as file:
                json.dump(self.__dict__, file)
        else:
            json.dump(self.__dict__, f)

    @staticmethod
    def load(f: Union[File, IO]) -> 'VideoMetadata':
        """
        Read the JSON metadata from disk.

        :param f: The file pointer or path to the read from.
        :return: The metadata object.
        """
        if isinstance(f, (str, Path)):
            with open(f) as file:
                kwargs = json.load(file)
        else:
            kwargs = json.load(f)

        return VideoMetadata(**kwargs)


class InvalidDatasetFormatError(Exception):
    """An error indicating that there is something wrong with the folder structure/files of a given dataset."""
    pass


class Dataset(abc.ABC):
    """The basic structure for datasets."""

    """The files required to be in the root folder of a dataset."""
    required_files = []

    """The folders required to be in the root folder of a dataset."""
    required_folders = []

    def __init__(self, base_path: File):
        """
        :param base_path: The path to the dataset.
        """
        self.base_path = base_path

        self.__class__._validate_dataset(base_path)

    @classmethod
    def is_valid_folder_structure(cls, path):
        """
        Check whether a folder conforms to the dataset format.

        :param path: The path to the dataset.
        :return: True if the path points to a valid dataset, False otherwise.
        """
        try:
            cls._validate_dataset(path)
            return True
        except InvalidDatasetFormatError:
            return False

    @classmethod
    def _validate_dataset(cls, base_path):
        """
        Check whether the given path points to a valid RGB-D dataset.

        This method will throw an AssertionError if the path does not point to a valid dataset.

        :param base_path: The path to the RGB-D dataset.
        """
        files_to_find = set(cls.required_files)
        folders_to_find = set(cls.required_folders)

        if not os.path.isdir(base_path):
            raise InvalidDatasetFormatError(f"The folder {base_path} does not exist!")

        for filename in os.listdir(base_path):
            file_path = pjoin(base_path, filename)

            if os.path.isfile(file_path):
                files_to_find.discard(filename)
            elif os.path.isdir(file_path):
                if len(os.listdir(file_path)) == 0 and filename in folders_to_find:
                    raise InvalidDatasetFormatError(f"Empty folder {filename} in {base_path}.")

                folders_to_find.discard(filename)

        if len(files_to_find) > 0:
            raise InvalidDatasetFormatError(
                f"Could not find the following required files {files_to_find} in {base_path}.")

        if len(folders_to_find) > 0:
            raise InvalidDatasetFormatError(
                f"Could not find the following required folders {folders_to_find} in {base_path}.")

    def __str__(self):
        return f"<{self.__class__.__name__} {Path(self.base_path).stem}>"


class DatasetMetadata:
    """Information about a dataset."""

    def __init__(self, num_frames: int, fps: float, width: int, height: int, estimate_pose: bool, estimate_depth: bool,
                 depth_mask_dilation_iterations: int, depth_scale: float, max_depth=10.0, frame_step=1,
                 colmap_options=COLMAPOptions()):
        """
        :param num_frames: The number of frames in the video sequence.
        :param fps: The framerate at which the video was captured.
        :param width: The width of a frame (pixels).
        :param height: The height of a frame (pixels).
        :param estimate_pose: Whether the camera parameters where estimated with COLMAP or from ground truth data.
        :param estimate_depth: Whether the depth maps where estimated or from ground truth data.
        :param depth_scale: A scalar that when multiplied with depth map, will transform the depth values to meters.
        :param max_depth: The maximum depth allowed in a depth map. Values exceeding this threshold will be set to zero.
        :param frame_step: The frequency that frames were sampled at for COLMAP and pose optimisation
            (only applicable if using estimated data).
        :param depth_mask_dilation_iterations: The number of times to apply the dilation filter to the dynamic object
            masks when creating the masked depth maps.
        """
        self.num_frames = num_frames
        self.fps = fps
        self.frame_step = frame_step
        self.width = width
        self.height = height
        self.depth_scale = depth_scale
        self.max_depth = max_depth
        self.depth_mask_dilation_iterations = depth_mask_dilation_iterations
        self.estimate_pose = estimate_pose
        self.estimate_depth = estimate_depth
        self.colmap_options = colmap_options

        if not isinstance(estimate_pose, bool):
            raise ValueError(f"`estimate_pose` must be a boolean, got {type(estimate_pose)}.")

        if not isinstance(estimate_depth, bool):
            raise ValueError(f"`estimate_depth` must be a boolean, got {type(estimate_depth)}.")

        check_domain(num_frames, 'num_frames', int, Domain.Positive)
        check_domain(frame_step, 'frame_step', int, Domain.Positive)
        check_domain(width, 'width', int, Domain.Positive)
        check_domain(height, 'height', int, Domain.Positive)
        check_domain(depth_scale, 'depth_scale', float)
        check_domain(max_depth, 'max_depth', float, Domain.Positive)
        check_domain(depth_mask_dilation_iterations, 'depth_mask_dilation_iterations', int, Domain.Positive)

    def __eq__(self, other: 'DatasetMetadata') -> bool:
        return self.num_frames == other.num_frames and \
            np.isclose(self.fps, other.fps) and \
            self.frame_step == other.frame_step and \
            self.width == other.width and \
            self.height == other.height and \
            np.isclose(self.depth_scale, other.depth_scale) and \
            np.isclose(self.max_depth, other.max_depth) and \
            self.depth_mask_dilation_iterations == other.depth_mask_dilation_iterations and \
            self.estimate_pose == other.estimate_pose and \
            self.estimate_depth == other.estimate_depth and \
            self.colmap_options == other.colmap_options

    def __repr__(self):
        return f"{self.__class__.__name__}(num_frames={self.num_frames}, fps={self.fps}, " \
               f"frame_step={self.frame_step}, width={self.width}, height={self.height}, " \
               f"max_depth={self.max_depth}, " \
               f"estimate_pose={self.estimate_pose}, " \
               f"estimate_depth={self.estimate_depth}, " \
               f"depth_mask_dilation_iterations={self.depth_mask_dilation_iterations}, " \
               f"depth_scale={self.depth_scale}, " \
               f"colmap_options={repr(self.colmap_options)})"

    def __str__(self):
        return f"Dataset info: {self.num_frames} frames, " \
               f"{self.width} x {self.height} @ {self.fps:.2f} fps with a duration of {self.duration}."

    @property
    def duration(self):
        """The length of the video sequence in seconds."""
        total_seconds = self.num_frames / self.fps

        return datetime.timedelta(seconds=total_seconds)

    def to_json(self) -> dict:
        """
        Convert the metadata to a JSON friendly dictionary.
        :return: A dictionary containing the metadata.
        """
        return dict(
            num_frames=self.num_frames,
            fps=self.fps,
            frame_step=self.frame_step,
            width=self.width,
            height=self.height,
            depth_scale=self.depth_scale,
            max_depth=self.max_depth,
            depth_mask_dilation_iterations=self.depth_mask_dilation_iterations,
            estimate_pose=self.estimate_pose,
            estimate_depth=self.estimate_depth,
            colmap_options=self.colmap_options.to_json()
        )

    @staticmethod
    def from_json(json_dict: dict) -> 'DatasetMetadata':
        """
        Get a dataset metadata from a JSON dictionary.

        :param json_dict: A JSON formatted dictionary.
        :return: The dataset metadata.
        """
        return DatasetMetadata(
            num_frames=int(json_dict['num_frames']),
            frame_step=int(json_dict['frame_step']),
            fps=float(json_dict['fps']),
            width=int(json_dict['width']),
            height=int(json_dict['height']),
            estimate_pose=bool(json_dict['estimate_pose']),
            estimate_depth=bool(json_dict['estimate_depth']),
            depth_scale=float(json_dict['depth_scale']),
            max_depth=float(json_dict['max_depth']),
            depth_mask_dilation_iterations=int(json_dict['depth_mask_dilation_iterations']),
            colmap_options=COLMAPOptions.from_json(json_dict['colmap_options'])
        )

    def save(self, f: Union[File, IO]):
        """
        Write the metadata to disk as a JSON file.

        :param f: The file pointer or path to write to.
        """
        if isinstance(f, (str, Path)):
            with open(f, 'w') as file:
                json.dump(self.to_json(), file)
        else:
            json.dump(self.to_json(), f)

    @staticmethod
    def load(f: Union[File, IO]) -> 'DatasetMetadata':
        """
        Read the JSON metadata from disk.

        :param f: The file pointer or path to the read from.
        :return: The metadata object.
        """
        if isinstance(f, (str, Path)):
            with open(f, 'r') as file:
                json_dict = json.load(file)
        else:
            json_dict = json.load(f)

        return DatasetMetadata.from_json(json_dict)


class HiveDataset(Dataset):
    """The main dataset format for the HIVE project."""

    metadata_filename = "metadata.json"
    camera_matrix_filename = "camera_matrix.txt"
    camera_trajectory_filename = "camera_trajectory.txt"

    required_files = [metadata_filename, camera_trajectory_filename, camera_matrix_filename]

    rgb_folder = "rgb"
    depth_folder = "depth"
    mask_folder = "mask"
    masked_depth_folder = 'masked_depth'

    inpainted_rgb_folder = f"{rgb_folder}_inpainted"
    inpainted_depth_folder = f"{depth_folder}_inpainted"
    inpainted_mask_folder = f"{mask_folder}_inpainted"

    required_folders = [rgb_folder, depth_folder, mask_folder]

    # Dataset adaptors are expected to convert depth maps to mm.
    # This scaling factor converts the mm depth values to meters.
    depth_scaling_factor = 1. / 1000.

    def __init__(self, base_path):
        """
        :param base_path: The path to the dataset.
        """
        super().__init__(base_path=base_path)

        self.metadata = DatasetMetadata.load(self.path_to_metadata)

        self.camera_matrix, self.camera_trajectory = self._load_camera_parameters()

        self.rgb_dataset = ImageFolderDataset(self.path_to_rgb_frames)
        self.depth_dataset = ImageFolderDataset(self.path_to_depth_maps, transform=self._get_depth_map_transform())
        self.mask_dataset = ImageFolderDataset(self.path_to_masks)

        self.inpainted_rgb_dataset, self.inpainted_depth_dataset = self._get_inpainted_frame_data()

        self._masked_depth_path: Optional[str] = None

    def _get_inpainted_frame_data(self) -> \
            Tuple[Optional[ImageFolderDataset], Optional[ImageFolderDataset]]:
        """
        Returns the inpainted RGB, depth and mask datasets, if they exist.
        """
        if not os.path.isdir(self.path_to_inpainted_rgb_frames) or \
                not os.path.isdir(self.path_to_inpainted_depth_maps) or \
                not os.path.isdir(self.path_to_inpainted_masks):
            return None, None

        inpainted_rgb_dataset = ImageFolderDataset(self.path_to_inpainted_rgb_frames)
        inpainted_depth_dataset = ImageFolderDataset(self.path_to_inpainted_depth_maps,
                                                     transform=self._get_depth_map_transform())

        num_frames = self.num_frames

        if len(inpainted_rgb_dataset) != num_frames or len(inpainted_rgb_dataset) != num_frames:
            raise RuntimeError(f"Expected inpainted frame data to have {num_frames} frames, "
                               f"but got {len(inpainted_rgb_dataset)} and {len(inpainted_depth_dataset)}")

        return inpainted_rgb_dataset, inpainted_depth_dataset

    @property
    def bg_rgb_dataset(self) -> ImageFolderDataset:
        """The RGB frames for the background. Will use inpainted frame data if available."""
        return self.inpainted_rgb_dataset or self.rgb_dataset

    @property
    def bg_depth_dataset(self) -> ImageFolderDataset:
        """The depth maps for the background. Will use inpainted frame data if available."""
        return self.inpainted_depth_dataset or self.depth_dataset

    @property
    def has_inpainted_frame_data(self) -> bool:
        return self.inpainted_rgb_dataset is not None and self.inpainted_depth_dataset is not None

    @property
    def path_to_metadata(self):
        return pjoin(self.base_path, self.metadata_filename)

    @property
    def path_to_camera_matrix(self):
        return pjoin(self.base_path, self.camera_matrix_filename)

    @property
    def path_to_camera_trajectory(self):
        return pjoin(self.base_path, self.camera_trajectory_filename)

    @property
    def path_to_rgb_frames(self):
        return pjoin(self.base_path, self.rgb_folder)

    @property
    def path_to_depth_maps(self):
        return pjoin(self.base_path, self.depth_folder)

    @property
    def path_to_masks(self):
        return pjoin(self.base_path, self.mask_folder)

    @property
    def path_to_inpainted_rgb_frames(self):
        return pjoin(self.base_path, self.inpainted_rgb_folder)

    @property
    def path_to_inpainted_depth_maps(self):
        return pjoin(self.base_path, self.inpainted_depth_folder)

    @property
    def path_to_inpainted_masks(self):
        return pjoin(self.base_path, self.inpainted_mask_folder)

    @property
    def num_frames(self) -> int:
        return self.metadata.num_frames

    @property
    def frame_width(self) -> int:
        return self.metadata.width

    @property
    def frame_height(self) -> int:
        return self.metadata.height

    @property
    def fps(self) -> float:
        return self.metadata.fps

    @property
    def masked_depth_path(self):
        if self._masked_depth_path:
            return self._masked_depth_path
        else:
            raise RuntimeError(f"Masked depth maps have not been created for this dataset yet. "
                               f"Please make sure you have called `.create_masked_depth()` beforehand.")

    @property
    def fx(self) -> float:
        return self.camera_matrix[0, 0]

    @property
    def fy(self) -> float:
        return self.camera_matrix[1, 1]

    @property
    def cx(self) -> float:
        return self.camera_matrix[0, 2]

    @property
    def cy(self) -> float:
        return self.camera_matrix[1, 2]

    @property
    def fov_x(self) -> float:
        """The horizontal field of view in degrees."""
        return float(np.rad2deg(2. * np.arctan2(self.frame_width, 2. * self.fx)))

    @property
    def fov_y(self) -> float:
        """The vertical field of view in degrees."""
        return float(np.rad2deg(2 * np.arctan2(self.frame_height, 2 * self.fy)))

    def __len__(self):
        return self.num_frames

    def _get_depth_map_transform(self):
        def transform(depth_map):
            depth_map = self.depth_scaling_factor * depth_map.astype(np.float32)
            depth_map[depth_map > self.metadata.max_depth] = 0.0

            return depth_map

        return transform

    def create_masked_depth(self, dilation_options=MaskDilationOptions(num_iterations=64)) -> 'HiveDataset':
        start = datetime.datetime.now()

        masked_depth_folder = self.masked_depth_folder
        masked_depth_path = pjoin(self.base_path, masked_depth_folder)

        if os.path.isdir(masked_depth_path) and len(os.listdir(masked_depth_path)) == len(self):
            is_mask_dilation_iterations_same = self.metadata.depth_mask_dilation_iterations == dilation_options.num_iterations

            if is_mask_dilation_iterations_same:
                logging.info(f"Found cached masked depth at {masked_depth_path}")
                self._masked_depth_path = masked_depth_path

                return self
            else:
                logging.warning(
                    f"Found cached masked depth maps but they were created with mask dilation iterations of "
                    f"{self.metadata.depth_mask_dilation_iterations} instead of the specified "
                    f"{dilation_options.num_iterations}. The old masked depth maps will be replaced.")

        logging.info(f"Creating masked depth maps at {masked_depth_path}")

        os.makedirs(masked_depth_path, exist_ok=True)

        def save_depth(i, depth_map, mask):
            binary_mask = mask > 0.0
            binary_mask = dilate_mask(binary_mask, dilation_options)

            depth_map[binary_mask] = 0.0
            depth_map = depth_map / self.depth_scaling_factor  # undo depth scaling done during loading.
            depth_map = depth_map.astype(np.uint16)
            output_path = pjoin(masked_depth_path, f"{i:06d}.png")
            imageio.imwrite(output_path, depth_map)

        def save_depth_wrapper(args):
            save_depth(*args)

        logging.info(f"Writing masked depth to {masked_depth_path}...")
        args = list(zip(range(len(self)), self.depth_dataset, self.mask_dataset))
        tqdm_imap(save_depth_wrapper, args)

        self.metadata.depth_mask_dilation_iterations = dilation_options.num_iterations
        self.metadata.save(self.path_to_metadata)
        logging.info(f"Update metadata")

        elapsed = datetime.datetime.now() - start

        logging.info(f"Created {len(os.listdir(masked_depth_path))} masked depth maps in {elapsed}")

        return self

    def _load_camera_parameters(self) -> Tuple[np.ndarray, Trajectory]:
        """
        Load the ground truth camera matrix and trajectory from disk.

        The camera matrix is expected to be saved as a 3x3 matrix in a file called "camera.txt".
        The camera trajectory is expected to be saved as a Nx7 matrix in a file called "trajectory.txt", where N is the
        number of frames in the sequence and each row is a quaternion rotation 'r' and translation vector 't'.
        """
        camera_matrix = np.loadtxt(self.path_to_camera_matrix, dtype=np.float32)
        camera_trajectory = Trajectory.load(self.path_to_camera_trajectory)

        if camera_matrix.shape != (3, 3):
            raise RuntimeError(f"Expected camera matrix to be a 3x3 matrix,"
                               f" but got {camera_matrix.shape} instead.")

        if len(camera_trajectory.shape) != 2 or camera_trajectory.shape[1] != 7:
            raise RuntimeError(f"Expected camera trajectory to be a Nx7 matrix,"
                               f" but got {camera_trajectory.shape} instead.")

        return camera_matrix, camera_trajectory

    @staticmethod
    def index_to_filename(index: int, file_extension="png") -> str:
        return f"{index:06d}.{file_extension}"

    def select_key_frames(self, threshold=0.3, frame_step=30) -> List[int]:
        """
        From a dataset select `key frames', frames with overlap less than the specified ratio.

        :param threshold: The maximum overlap ratio before a frame is excluded from the key frame set. Note that
            setting the threshold to 1 will return the set of all frames and a threshold of 0 will return just the
            first frame.
        :param frame_step: The interval to sample frames at. A frame step of one will check every frame.
        :return: A set of key frames.
        """
        logging.info(f"Selecting key frames (threshold={threshold})...")

        if not (0.0 <= threshold <= 1.0):
            raise ValueError(f"Threshold must be a real number between zero and one (inclusive), but got {threshold}.")

        if threshold == 0.0:
            return [0]
        elif threshold == 1.0:
            return list(range(self.num_frames))

        if threshold > 0.8:
            logging.warning(f"Setting the key frame threshold to a high value (> 0.8) may result in long runtimes.")

        if frame_step < 1:
            raise ValueError(f"Frame step must be a positive integer, but got {frame_step} instead.")

        width = self.metadata.width
        height = self.metadata.height
        camera_matrix = self.camera_matrix

        key_frames = [0]

        for frame in tqdm(range(1, self.num_frames, frame_step)):
            depth = self.bg_depth_dataset[frame]
            mask = self.mask_dataset[frame] == 0
            pose = self.camera_trajectory[frame]
            R, t = get_pose_components(pose_vec2mat(pose))

            frame_points = point_cloud_from_depth(depth, mask, K=camera_matrix, R=R, t=t)

            for key_frame in key_frames:
                pose_key_frame = self.camera_trajectory[key_frame]
                R, t = get_pose_components(pose_vec2mat(pose_key_frame))

                points_projected_onto_key_frame, _ = world2image(frame_points, K=camera_matrix, R=R, t=t)

                valid_points_x = (points_projected_onto_key_frame[:, 0] >= 0) & (
                        points_projected_onto_key_frame[:, 0] < width)
                valid_points_y = (points_projected_onto_key_frame[:, 1] >= 0) & (
                        points_projected_onto_key_frame[:, 1] < height)
                visible_points_mask = valid_points_x & valid_points_y
                visible_points = points_projected_onto_key_frame[visible_points_mask]

                if len(visible_points) == 0:
                    continue

                min_extents = np.min(visible_points, axis=0)
                max_extents = np.max(visible_points, axis=0)
                visible_area = np.product(max_extents - min_extents)
                overlap_ratio = visible_area / (width * height)

                if overlap_ratio >= threshold:
                    logging.debug(f"Excluding frame {frame} from key frames. Reason: The overlap between frame {frame} "
                                  f"and the key frame {key_frame} was {overlap_ratio:.2f}, above the threshold of "
                                  f"{threshold:.2f}.")
                    break
            else:
                logging.debug(f"Adding frame {frame} to key frames.")
                key_frames.append(frame)

        logging.debug(f"Selected key frames: {key_frames}.")

        return key_frames


@contextlib.contextmanager
def temporary_trajectory(dataset: HiveDataset, trajectory: Trajectory):
    """
    Context manager that temporarily replaces the trajectory of a dataset.

    :param dataset: The dataset.
    :param trajectory: The trajectory to use.
    """
    traj_backup = dataset.camera_trajectory.copy()

    try:
        dataset.camera_trajectory = trajectory

        yield
    finally:
        dataset.camera_trajectory = traj_backup
