import datetime
import json
import os
import re
import shutil
import struct
import subprocess
import warnings
from argparse import Namespace
from collections import OrderedDict
from multiprocessing.pool import ThreadPool
from pathlib import Path
from typing import Union, Tuple, Optional, Callable, IO

import cv2
import imageio
import numpy as np
import psutil
import torch
from PIL import Image
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.engine import DefaultPredictor
from matplotlib import pyplot as plt
from scipy.spatial.transform import Rotation
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from Video2mesh.geometry import pose_vec2mat, dilate_mask
from Video2mesh.options import COLMAPOptions, MaskDilationOptions, DepthEstimationModel, DepthOptions
from Video2mesh.utils import log
from thirdparty.AdaBins.infer import InferenceHelper
from thirdparty.colmap.scripts.python.read_write_model import read_model

File = Union[str, Path]
Size = Tuple[int, int]


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
        buffersize = max(16 * 1024 ** 2 // image.itemsize, 1)

        for chunk in np.nditer(
                float32_image,
                flags=["external_loop", "buffered", "zerosize_ok"],
                buffersize=buffersize,
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


def create_masks(rgb_loader: DataLoader, mask_folder: Union[str, Path],
                 overwrite_ok=False, for_colmap=False):
    """
    Create instance segmentation masks for the given RGB video sequence and save the masks to disk..

    :param rgb_loader: The PyTorch DataLoader that loads the RGB frames (no data augmentations applied).
    :param mask_folder: The path to save the masks to.
    :param overwrite_ok: Whether it is okay to write over any mask files in `mask_folder` if it already exists.
    :param for_colmap: Whether the masks are intended for use with COLMAP or 3D video generation.
        Masks will be black and white with the background coloured white and using the
        corresponding input image's filename.
    """
    print(f"Creating masks...")

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
    print(class_names)

    person_label = class_names.index('person')
    predictor = BatchPredictor(cfg)

    os.makedirs(mask_folder, exist_ok=overwrite_ok)
    i = 0

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

            if for_colmap:
                output_filename = f"{rgb_loader.dataset.image_filenames[i]}.png"
            else:
                output_filename = f"{i:06d}.png"

            Image.fromarray(combined_masks).convert('L').save(os.path.join(mask_folder, output_filename))

            i += 1

        log(f"[{i:03,d}/{len(rgb_loader.dataset):03,d}] Creating segmentation masks...", prefix='\r', end='')

    print()


class COLMAPProcessor:
    """
    Estimates camera trajectory and intrinsic parameters via COLMAP.
    """

    def __init__(self, image_path: File, workspace_path: File, colmap_options: COLMAPOptions,
                 colmap_mask_folder='masks'):
        self.image_path = image_path
        self.workspace_path = workspace_path
        self.colmap_options = colmap_options
        self.mask_folder = colmap_mask_folder

    @property
    def mask_path(self):
        return os.path.join(self.workspace_path, self.mask_folder)

    @property
    def result_path(self):
        return os.path.join(self.workspace_path, 'sparse')

    @property
    def probably_has_results(self):
        recon_result_path = os.path.join(self.result_path, '0')
        min_files_for_recon = 4

        return os.path.isdir(self.result_path) and len(os.listdir(self.result_path)) > 0 and \
               (os.path.isdir(recon_result_path) and len(os.listdir(recon_result_path)) >= min_files_for_recon)

    def run(self):
        os.makedirs(self.workspace_path, exist_ok=True)

        if not os.path.isdir(self.mask_path) or len(os.listdir(self.mask_path)) == 0:
            print(f"Could not find masks in folder: {self.mask_path}.")
            print(f"Creating masks for COLMAP...")
            rgb_loader = DataLoader(ImageFolderDataset(self.image_path), batch_size=8, shuffle=False)
            create_masks(rgb_loader, self.mask_path, overwrite_ok=True, for_colmap=True)
        else:
            print(f"Found {len(os.listdir(self.mask_path))} masks in {self.mask_path}.")

        command = self.get_command()
        # TODO: Check that COLMAP is using GPU
        colmap_process = subprocess.Popen(command)
        colmap_process.wait()

        if colmap_process.returncode != 0:
            raise RuntimeError(f"COLMAP exited with code {colmap_process.returncode}")

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
        num_models = len(os.listdir(self.result_path))
        assert num_models == 1, f"COLMAP reconstructed {num_models} models when 1 was expected."
        sparse_recon_path = os.path.join(self.result_path, '0')

        log(f"Reading COLMAP model from {sparse_recon_path}...")
        cameras, images, points3d = read_model(sparse_recon_path, ext=".bin")

        return cameras, images, points3d

    def load_camera_params(self, raw_pose: Optional[bool] = None):
        """
        Load the camera intrinsic and extrinsic parameters from a COLMAP sparse reconstruction model.
        :param raw_pose: (optional) Whether to use the raw pose data straight from COLMAP.
                         This value will override `COLMAPProcessor.colmap_options.use_raw_pose` if set.
        :return: A 2-tuple containing the camera matrix (intrinsic parameters) and camera trajectory (extrinsic
            parameters). Each row in the camera trajectory consists of the rotation as a quaternion and the
            translation vector, in that order.
        """
        if raw_pose is None:
            raw_pose = self.colmap_options.use_raw_pose

        cameras, images, points3d = self._load_model()

        f, cx, cy, _ = cameras[1].params  # cameras is a dict, COLMAP indices start from one.

        intrinsic = np.eye(3)
        intrinsic[0, 0] = f
        intrinsic[1, 1] = f
        intrinsic[0, 2] = cx
        intrinsic[1, 2] = cy
        print("Read intrinsic parameters.")

        extrinsic = []

        if raw_pose:
            for image in images.values():
                # COLMAP quaternions seem to be stored in scalar first format.
                # However, rather than assuming the format we can just rely on the provided function to convert to a
                # rotation matrix, and use SciPy to convert that to a quaternion in scalar last format.
                # This avoids any future issues if the quaternion format ever changes.
                r = Rotation.from_matrix(image.qvec2rotmat()).as_quat()
                t = image.tvec

                extrinsic.append(np.hstack((r, t)))
        else:
            # Code adapted from https://github.com/facebookresearch/consistent_depth
            # According to some comments in the above code, "Note that colmap uses a different coordinate system
            # where y points down and z points to the world." The below rotation apparently puts the poses back into
            # a 'normal' coordinate frame.
            colmap_to_normal = np.diag([1, -1, -1])

            for image in images.values():
                R = image.qvec2rotmat()
                t = image.tvec.reshape(-1, 1)

                R, t = R.T, -R.T.dot(t)
                R = colmap_to_normal.dot(R).dot(colmap_to_normal.T)
                t = colmap_to_normal.dot(t)
                t = t.squeeze()

                r = Rotation.from_matrix(R).as_quat()

                extrinsic.append(np.hstack((r, t)))

        extrinsic = np.asarray(extrinsic).squeeze()

        print(f"Read extrinsic parameters for {len(extrinsic)} frames.")

        return intrinsic, extrinsic

    def get_sparse_depth_maps(self):
        cameras, images, points3d = self._load_model()

        camera_matrix, camera_trajectory = self.load_camera_params()

        source_image_shape = cv2.imread(os.path.join(self.image_path, images[1].name)).shape[:2]
        depth_maps = np.zeros((len(images), *source_image_shape), dtype=np.float32)

        for index_zero_based in range(len(images)):
            index = index_zero_based + 1
            image_data = images[index]

            points2d = np.round(image_data.xys[:, ::-1]).astype(int)

            K = np.eye(4)
            K[:3, :3] = camera_matrix
            points = np.zeros((len(points2d), 3))
            has_3d_points = np.zeros(len(points2d), dtype=bool)

            for i, point3d_id in enumerate(image_data.point3D_ids):
                if point3d_id == -1:
                    continue

                points[i] = points3d[point3d_id].xyz
                has_3d_points[i] = True

            pose = pose_vec2mat(camera_trajectory[index_zero_based])
            R = pose[:3, :3]
            t = pose[:3, 3:]

            points_camera_space = (R @ points.T) + t
            projected_points = (camera_matrix @ points_camera_space).T
            projected_points[~has_3d_points] = [0., 0., 0.]

            depth_maps[index_zero_based, points2d[:, 0], points2d[:, 1]] = projected_points[:, 2]

        return depth_maps


class NumpyDataset(Dataset):
    def __init__(self, rgb_frames):
        self.frames = rgb_frames

    def __getitem__(self, index):
        return self.frames[index]

    def __len__(self):
        return len(self.frames)


class ImageFolderDataset(Dataset):
    def __init__(self, base_dir, transform=None):
        assert os.path.isdir(base_dir), f"Could not find the folder: {base_dir}"

        self.base_dir = base_dir
        self.transform = transform

        filenames = list(sorted(os.listdir(base_dir)))
        assert len(filenames) > 0, f"No files found in the folder: {base_dir}"

        self.image_filenames = filenames
        self.image_paths = [os.path.join(base_dir, filename) for filename in filenames]

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]

        if image_path.endswith('.raw'):
            image = load_raw_float32_image(image_path)
        else:
            image = Image.open(image_path)

            if image.mode == 'I':
                image = image.convert('I;16')
            elif image.mode != 'L' and image.mode != 'I;16':
                image = image.convert('RGB')

            image = np.asarray(image)

        if self.transform:
            image = self.transform(image)

        return image

    def __len__(self):
        return len(self.image_paths)


class FrameSampler:
    """
    Samples a subset of frames.
    """

    def __init__(self, start=0, stop=-1, step=1, fps=30.0, stop_is_inclusive=False):
        """
        :param start: The index of the first frame to sample.
        :param stop: The index of the last frame to index. Setting this to `-1` is equivalent to setting it to the index
            of the last frame.
        :param step: The gap between each of the selected frame indices.
        :param fps: The frame rate of the video that is being sampled. Important if you want to sample frames based on
            time based figures.
        :param stop_is_inclusive: Whether to sample frames as an open range (`stop` is not included) or a closed range
            (`stop` is included).
        """
        self.start = start
        self.stop = stop
        self.step = step
        self.fps = fps

        self.stop_is_inclusive = stop_is_inclusive

    def __repr__(self):
        kv_pairs = map(lambda kv: "%s=%s" % kv, self.__dict__.items())

        return "<%s(%s)>" % (self.__class__.__name__, ', '.join(kv_pairs))

    def frame_range(self, start, stop=-1):
        """
        Select a range of frames.
        :param start: The index of the first frame to sample (inclusive).
        :param stop: The index of the last frame to sample (inclusive only if `stop_is_inclusive` is set to `True`).
        :return: A new FrameSampler with the new frame range.
        """
        options = dict(self.__dict__)
        options.update(start=start, stop=stop)

        return FrameSampler(**options)

    def frame_interval(self, step):
        """
        Choose the frequency at which frames are sampled.
        :param step: The integer gap between sampled frames.
        :return: A new FrameSampler with the new sampling frequency.
        """
        options = dict(self.__dict__)
        options.update(step=step)

        return FrameSampler(**options)

    def time_range(self, start, stop=None):
        """
        Select a range of frames based on time.
        :param start: The time of the first frame to sample (in seconds, inclusive).
        :param stop: The time of the last frame to sample (in seconds, inclusive only if `stop_is_inclusive` is set to
            `True`).
        :return: A new FrameSampler with the new frame range.
        """
        options = dict(self.__dict__)

        start_frame = int(start * self.fps)

        if stop:
            stop_frame = int(stop * self.fps)
        else:
            stop_frame = -1

        options.update(start=start_frame, stop=stop_frame)

        return FrameSampler(**options)

    def time_interval(self, step):
        """
        Choose the frequency at which frames are sampled.
        :param step: The time (in seconds) between sampled frames.
        :return: A new FrameSampler with the new sampling frequency.
        """
        options = dict(self.__dict__)

        frame_step = int(step * self.fps)

        options.update(step=frame_step)

        return FrameSampler(**options)

    def choose(self, frames):
        """
        Choose frames based on the sampling range and frequency defined in this object.
        :param frames: The frames to sample from.
        :return: The subset of sampled frames.
        """
        num_frames = len(frames[0])

        if self.stop < 0:
            stop = num_frames
        else:
            stop = self.stop

        if self.stop_is_inclusive:
            stop += self.step

        rgb = frames[0][self.start:stop:self.step]
        depth = frames[1][self.start:stop:self.step]
        pose = frames[2][self.start:stop:self.step]
        return rgb, depth, pose


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


class DatasetBase:
    """The basic structure for datasets."""

    """The files required to be in the root folder of a dataset."""
    required_files = []

    """The folders required to be in the root folder of a dataset."""
    required_folders = []

    def __init__(self, base_path: File, overwrite_ok=False):
        """
        :param base_path: The path to the dataset.
        :param overwrite_ok: Whether it is okay to overwrite existing adapted dataset.
        """
        self.base_path = base_path
        self.overwrite_ok = overwrite_ok

        self._validate_dataset(base_path)

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
            file_path = os.path.join(base_path, filename)

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

    def __init__(self, num_frames: int, fps: float, width: int, height: int, depth_scale: float, max_depth=10.0,
                 depth_mask_dilation_iterations: Optional[int] = None, depth_estimation_model: Optional[str] = None):
        """
        :param num_frames: The number of frames in the video sequence.
        :param fps: The framerate at which the video was captured.
        :param width: The width of a frame (pixels).
        :param height: The height of a frame (pixels).
        :param depth_scale: A scalar that when multiplied with depth map, will transform the depth values to meters.
        :param max_depth: The maximum depth allowed in a depth map. Values exceeding this threshold will be set to zero.
        :param depth_mask_dilation_iterations: The number of times to apply the dilation filter to the dynamic object
            masks when creating the masked depth maps.
        :param depth_estimation_model: The name of the depth estimation model used to create the estimated depth maps.
        """
        self.num_frames = num_frames
        self.fps = fps
        self.width = width
        self.height = height
        self.depth_scale = depth_scale
        self.max_depth = max_depth
        self.depth_mask_dilation_iterations = depth_mask_dilation_iterations
        self.depth_estimation_model = depth_estimation_model

        if not isinstance(num_frames, int) or num_frames < 1:
            raise ValueError(f"Num frames must be a positive integer, got {num_frames}.")

        if not isinstance(width, int) or width < 1:
            raise ValueError(f"Width must be a positive integer, got {width}.")

        if not isinstance(height, int) or height < 1:
            raise ValueError(f"Height must be a positive integer, got {height}.")

        if not isinstance(depth_scale, (float, int)) or not np.isfinite(depth_scale) or depth_scale <= 0.0:
            raise ValueError(f"Depth scale must be a positive, finite number, but got {depth_scale}.")

        if not isinstance(depth_mask_dilation_iterations, int) or depth_mask_dilation_iterations < 0:
            raise ValueError(f"Depth mask dilation iterations must be a non-negative integer, "
                             f"but got {depth_mask_dilation_iterations}.")

        if depth_estimation_model is not None and depth_estimation_model not in DepthEstimationModel.get_choices():
            raise ValueError(f"Invalid depth estimation model. "
                             f"Got {depth_estimation_model} but expected one of the following: "
                             f"{list(DepthEstimationModel.get_choices().keys())}")

    def __repr__(self):
        return f"{self.__class__.__name__}(num_frames={self.num_frames}, fps={self.fps}, " \
               f"depth_scale={self.depth_scale}, width={self.width}, height={self.height}," \
               f"depth_mask_dilation_iterations={self.depth_mask_dilation_iterations}," \
               f"depth_estimation_model={self.depth_estimation_model})"

    def __str__(self):
        return f"Dataset info: {self.num_frames} frames, " \
               f"{self.width} x {self.height} @ {self.fps:.2f} fps with a duration of {self.duration}."

    @property
    def duration(self):
        """The length of the video sequence in seconds."""
        total_seconds = self.num_frames / self.fps

        return datetime.timedelta(seconds=total_seconds)

    def save(self, f: Union[File, IO]):
        """
        Write the metadata to disk as a JSON file.

        :param f: The file pointer or path to write to.
        """
        if isinstance(f, (str, Path)):
            with open(f, 'w') as file:
                json.dump(self.__dict__, file)
        else:
            json.dump(self.__dict__, f)

    @staticmethod
    def load(f: Union[File, IO]) -> 'DatasetMetadata':
        """
        Read the JSON metadata from disk.

        :param f: The file pointer or path to the read from.
        :return: The metadata object.
        """
        if isinstance(f, (str, Path)):
            with open(f, 'r') as file:
                kwargs = json.load(file)
        else:
            kwargs = json.load(f)

        return DatasetMetadata(**kwargs)


class VTMDataset(DatasetBase):
    """The main dataset format for the video2mesh (VTM) project."""

    metadata_filename = "metadata.json"
    camera_matrix_filename = "camera_matrix.txt"
    camera_trajectory_filename = "camera_trajectory.txt"
    estimated_camera_matrix_filename = "estimated_camera_matrix.txt"
    estimated_camera_trajectory_filename = "estimated_camera_trajectory.txt"

    required_files = [camera_trajectory_filename, camera_matrix_filename]

    rgb_folder = "rgb"
    depth_folder = "depth"
    estimated_depth_folder = "estimated_depth"
    mask_folder = "mask"
    colmap_folder = "colmap"

    required_folders = [rgb_folder, depth_folder]

    def __init__(self, base_path, overwrite_ok=False):
        """
        :param base_path: The path to the dataset.
        :param overwrite_ok: Whether it is okay to overwrite existing adapted dataset.
        """
        super().__init__(base_path=base_path, overwrite_ok=overwrite_ok)

        self._using_estimated_depth = False
        self._using_estimated_camera_parameters = False

        self.metadata = DatasetMetadata.load(self.path_to_metadata)

        self.camera_matrix, self.camera_trajectory = self._load_camera_parameters()

        self.rgb_dataset = ImageFolderDataset(self.path_to_rgb_frames)
        self.depth_dataset = ImageFolderDataset(self.path_to_depth_maps, transform=self._get_depth_map_transform())
        self._mask_dataset: Optional[ImageFolderDataset] = None

        self._masked_depth_path: Optional[str] = None

    @property
    def path_to_metadata(self):
        return os.path.join(self.base_path, self.metadata_filename)

    @property
    def path_to_camera_matrix(self):
        return os.path.join(self.base_path, self.camera_matrix_filename)

    @property
    def path_to_camera_trajectory(self):
        return os.path.join(self.base_path, self.camera_trajectory_filename)

    @property
    def path_to_estimated_camera_matrix(self):
        return os.path.join(self.base_path, self.estimated_camera_matrix_filename)

    @property
    def path_to_estimated_camera_trajectory(self):
        return os.path.join(self.base_path, self.estimated_camera_trajectory_filename)

    @property
    def path_to_rgb_frames(self):
        return os.path.join(self.base_path, self.rgb_folder)

    @property
    def path_to_depth_maps(self):
        return os.path.join(self.base_path, self.depth_folder)

    @property
    def path_to_estimated_depth_maps(self):
        return os.path.join(self.base_path, self.estimated_depth_folder)

    @property
    def path_to_masks(self):
        return os.path.join(self.base_path, self.mask_folder)

    @property
    def path_to_colmap(self):
        return os.path.join(self.base_path, self.colmap_folder)

    @property
    def num_frames(self):
        return self.metadata.num_frames

    @property
    def frame_width(self):
        return self.metadata.width

    @property
    def frame_height(self):
        return self.metadata.height

    @property
    def mask_dataset(self):
        if self._mask_dataset:
            return self._mask_dataset
        else:
            raise RuntimeError(f"Masks have not been created for this dataset yet. "
                               f"Please make sure you have called `.create_masks()` before trying to access the masks.")

    @property
    def masked_depth_path(self):
        if self._masked_depth_path:
            return self._masked_depth_path
        else:
            raise RuntimeError(f"Masked depth maps have not been created for this dataset yet. "
                               f"Please make sure you have called `.create_masked_depth()` beforehand.")

    @property
    def masked_depth_folder(self):
        if self._using_estimated_depth:
            return 'masked_estimated_depth'
        else:
            return 'masked_depth'

    @property
    def depth_scaling_factor(self):
        if self._using_estimated_depth:
            return 1. / 1000.
        else:
            return self.metadata.depth_scale

    @property
    def fx(self):
        return self.camera_matrix[0, 0]

    @property
    def fy(self):
        return self.camera_matrix[1, 1]

    @property
    def cx(self):
        return self.camera_matrix[0, 2]

    @property
    def cy(self):
        return self.camera_matrix[1, 2]

    def __len__(self):
        return self.num_frames

    def _get_depth_map_transform(self):
        def transform(depth_map):
            depth_map = self.depth_scaling_factor * depth_map.astype(np.float32)
            depth_map[depth_map > self.metadata.max_depth] = 0.0

            return depth_map

        return transform

    def create_or_find_masks(self) -> 'VTMDataset':
        """
        Locate the instance segmentation masks (if they exist), otherwise create them.
        """
        if os.path.isdir(self.path_to_masks) and len(os.listdir(self.path_to_masks)) == self.num_frames:
            log(f"Found cached masks.")
        else:
            rgb_loader = DataLoader(self.rgb_dataset, batch_size=8, shuffle=False)
            create_masks(rgb_loader=rgb_loader, mask_folder=self.path_to_masks, overwrite_ok=self.overwrite_ok)

        self._mask_dataset = ImageFolderDataset(self.path_to_masks)

        first_mask = self._mask_dataset[0]
        height, width = first_mask.shape[:2]

        expected_width = self.frame_width
        expected_height = self.frame_height

        if height != expected_height or width != expected_width:
            raise RuntimeError(f"Expected masks with a resolution of {expected_width}x{expected_height}, "
                               f"but got {width}x{height} (width, height).")

        return self

    def create_masked_depth(self, dilation_options=MaskDilationOptions(num_iterations=64)) -> 'VTMDataset':
        start = datetime.datetime.now()

        masked_depth_folder = self.masked_depth_folder
        masked_depth_path = os.path.join(self.base_path, masked_depth_folder)

        if os.path.isdir(masked_depth_path) and len(os.listdir(masked_depth_path)) == len(self):
            is_mask_dilation_iterations_same = self.metadata.depth_mask_dilation_iterations == dilation_options.num_iterations

            if is_mask_dilation_iterations_same:
                log(f"Found cached masked depth at {masked_depth_path}")
                self._masked_depth_path = masked_depth_path

                return self
            else:
                warnings.warn(f"Found cached masked depth maps but they were created with mask dilation iterations of "
                              f"{self.metadata.depth_mask_dilation_iterations} instead of the specified "
                              f"{dilation_options.num_iterations}. The old masked depth maps will be replaced.")

        log(f"Creating masked depth maps at {masked_depth_path}")

        os.makedirs(masked_depth_path, exist_ok=True)
        log("Create output folder")

        pool = ThreadPool(processes=psutil.cpu_count(logical=False))
        log("Create thread pool")

        def save_depth(i, depth_map, mask):
            binary_mask = mask > 0.0
            binary_mask = dilate_mask(binary_mask, dilation_options)

            depth_map[binary_mask] = 0.0
            depth_map = depth_map / self.depth_scaling_factor  # undo depth scaling done during loading.
            depth_map = depth_map.astype(np.uint16)
            output_path = os.path.join(masked_depth_path, f"{i:06d}.png")
            imageio.imwrite(output_path, depth_map)

            log(f"Writing masked depth to {output_path}")

        pool.starmap(save_depth, zip(range(len(self)), self.depth_dataset, self.mask_dataset))

        self.metadata.depth_mask_dilation_iterations = dilation_options.num_iterations
        self.metadata.save(self.path_to_metadata)
        log(f"Update metadata")

        elapsed = datetime.datetime.now() - start

        log(f"Created {len(os.listdir(masked_depth_path))} masked depth maps in {elapsed}")

        return self

    def use_ground_truth_depth(self) -> 'VTMDataset':
        self.depth_dataset = ImageFolderDataset(self.path_to_depth_maps, transform=self._get_depth_map_transform())
        self._using_estimated_depth = False
        self._masked_depth_path = None

        return self

    def use_estimated_depth(self, depth_options=DepthOptions()) -> 'VTMDataset':
        """Use estimated depth maps, or create them if they do not already exist."""
        estimated_depth_path = self.path_to_estimated_depth_maps
        depth_estimation_model = depth_options.depth_estimation_model

        uses_same_model = self.metadata.depth_estimation_model == depth_estimation_model.name.lower()

        if os.path.isdir(estimated_depth_path) and uses_same_model:
            depth_dataset = ImageFolderDataset(estimated_depth_path, transform=self._get_depth_map_transform())
            num_estimated_depth_maps = len(depth_dataset)
            num_frames = self.num_frames

            if num_estimated_depth_maps != num_frames:
                raise RuntimeError(f"Found estimated depth maps in {estimated_depth_path} but found "
                                   f"{num_estimated_depth_maps} when {num_frames} was expected. "
                                   f"Potential fix: Delete the folder {estimated_depth_path} and run the program "
                                   f"again.")

            first_depth_map = depth_dataset[0]
            height, width = first_depth_map.shape[:2]
            expected_height = self.frame_height
            expected_width = self.frame_width

            if height != expected_height or width != expected_width:
                raise RuntimeError(f"Found estimated depth maps in {estimated_depth_path} but found "
                                   f"depth maps of size {width}x{height} when {expected_width}x{expected_height} "
                                   f"(width, height) was expected. "
                                   f"Potential fix: Delete the folder {estimated_depth_path} and run the program "
                                   f"again.")

            log(f"Found estimated depth maps in {estimated_depth_path}.")
        else:
            if os.path.isdir(estimated_depth_path) and not uses_same_model:
                warnings.warn(f"Found cached depth maps but they were created with the depth estimation model "
                              f"{self.metadata.depth_estimation_model} instead of the specified "
                              f"{depth_estimation_model.name.lower()}. The old depth maps will be replaced.")

            log("Estimating depth maps...")

            if depth_estimation_model == DepthEstimationModel.ADABINS:
                # TODO: Get the weights path from a config file pointing to the location in the Docker image.
                adabins_inference = InferenceHelper(weights_path='/root/.cache/pretrained')
                adabins_inference.predict_dir(self.path_to_rgb_frames, out_dir=estimated_depth_path)
            elif depth_estimation_model == DepthEstimationModel.LERES:
                self._estimate_depth_leres(estimated_depth_path)
            else:
                raise RuntimeError(f"Unsupported depth estimation model '{depth_estimation_model.name}'.")

            depth_dataset = ImageFolderDataset(estimated_depth_path, transform=self._get_depth_map_transform())

            self.metadata.depth_estimation_model = depth_estimation_model.name.lower()
            self.metadata.save(self.path_to_metadata)
            log(f"Update metadata")

        self._masked_depth_path = None
        self.depth_dataset = depth_dataset
        self._using_estimated_depth = True

        return self

    def _estimate_depth_leres(self, estimated_depth_path: str):
        # TODO: Make the below options configurable via cli.
        args = Namespace()
        args.image_path = self.path_to_rgb_frames
        args.output_path = estimated_depth_path
        args.load_ckpt = os.path.join(os.environ['WEIGHTS_PATH'], 'res101.pth')
        args.backbone = 'resnext101'

        # create depth model
        from thirdparty.adelai_depth.LeReS.lib.net_tools import load_ckpt
        from thirdparty.adelai_depth.LeReS.lib.multi_depth_model_woauxi import RelDepthModel

        depth_model = RelDepthModel(backbone=args.backbone)
        depth_model.eval()

        # load checkpoint
        load_ckpt(args, depth_model, None, None)
        depth_model.cuda()

        images_folder = args.image_path
        imgs_list = os.listdir(images_folder)
        imgs_list.sort()
        imgs_path = [os.path.join(images_folder, i) for i in imgs_list if i != 'outputs']
        image_dir_out = args.output_path
        os.makedirs(image_dir_out, exist_ok=True)

        def scale_torch(img):
            """
            Scale the image and output it in torch.tensor.
            :param img: input rgb is in shape [H, W, C], input depth/disp is in shape [H, W]
            :param scale: the scale factor. float
            :return: img. [C, H, W]
            """
            if len(img.shape) == 2:
                img = img[np.newaxis, :, :]

            if img.shape[2] == 3:
                transform = transforms.Compose([transforms.ToTensor(),
                                                transforms.Normalize((0.485, 0.456, 0.406),
                                                                     (0.229, 0.224, 0.225))])
                img = transform(img)
            else:
                img = img.astype(np.float32)
                img = torch.from_numpy(img)

            return img

        for i, image_path in enumerate(imgs_path):
            print(f"[{i:04d}/{len(imgs_path):04d}] {image_path}")
            rgb = cv2.imread(image_path)
            rgb_c = rgb[:, :, ::-1].copy()
            A_resize = cv2.resize(rgb_c, (448, 448))

            img_torch = scale_torch(A_resize)[None, :, :, :]
            pred_depth = depth_model.inference(img_torch).cpu().numpy().squeeze()
            pred_depth_ori = cv2.resize(pred_depth, (rgb.shape[1], rgb.shape[0]))

            if pred_depth_ori.max() > 10.0:
                warnings.warn("Found depth value greater than 10.0.")

            pred_depth_ori[pred_depth_ori > 10.0] = 10.0
            pred_depth_mm = (1000 * pred_depth_ori).astype(
                np.uint16)  # Convert to mm as per KinectFusion style datasets.

            imageio.imwrite(os.path.join(image_dir_out, f"{i:06d}.png"), pred_depth_mm)

    def use_ground_truth_camera_parameters(self) -> 'VTMDataset':
        self.camera_matrix, self.camera_trajectory = self._load_camera_parameters(load_ground_truth_data=True)
        self._using_estimated_camera_parameters = False

        return self

    def use_estimated_camera_parameters(self, colmap_options: COLMAPOptions) -> 'VTMDataset':
        """
        Use camera matrix and trajectory data estimated with COLMAP.
        These will be created if they do not already exist.
        """
        # TODO: Sample subset of frames to speed up this step.
        processor = self._get_colmap_processor(colmap_options)

        if not processor.probably_has_results:
            processor.run()

            camera_matrix, _ = processor.load_camera_params()
            camera_trajectory = self._adjust_colmap_poses(colmap_options)

            np.savetxt(self.path_to_estimated_camera_matrix, camera_matrix)
            np.savetxt(self.path_to_estimated_camera_trajectory, camera_trajectory)

        self.camera_matrix, self.camera_trajectory = self._load_camera_parameters(load_ground_truth_data=False)
        self._using_estimated_camera_parameters = True

        return self

    def _load_camera_parameters(self, load_ground_truth_data=True) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load the ground truth camera matrix and trajectory from disk.

        The camera matrix is expected to be saved as a 3x3 matrix in a file called "camera.txt".
        The camera trajectory is expected to be saved as a Nx7 matrix in a file called "trajectory.txt", where N is the
        number of frames in the sequence and each row is a quaternion rotation 'r' and translation vector 't'.

        :param load_ground_truth_data: Whether to use the ground truth camera parameters or to use
            estimated camera parameters (e.g. from COLMAP).
        """
        if load_ground_truth_data:
            camera_matrix = np.loadtxt(self.path_to_camera_matrix)
            camera_trajectory = np.loadtxt(self.path_to_camera_trajectory)
        else:
            if not os.path.isfile(self.path_to_estimated_camera_matrix) or \
                    not os.path.isfile(self.path_to_estimated_camera_trajectory):
                raise RuntimeError(f"Could not find either/both "
                                   f"the camera matrix file at {self.estimated_camera_matrix_filename} and "
                                   f"the camera trajectory file at {self.path_to_estimated_camera_trajectory}. "
                                   f"Make sure you have run `VTMDataset(...).use_estimated_camera_params().")

            camera_matrix = np.loadtxt(self.path_to_estimated_camera_matrix)
            camera_trajectory = np.loadtxt(self.path_to_estimated_camera_trajectory)

        if camera_matrix.shape != (3, 3):
            raise RuntimeError(f"Expected camera matrix to be a 3x3 matrix,"
                               f" but got {camera_matrix.shape} instead.")

        if len(camera_trajectory.shape) != 2 or camera_trajectory.shape[1] != 7:
            raise RuntimeError(f"Expected camera trajectory to be a Nx7 matrix,"
                               f" but got {camera_trajectory.shape} instead.")

        return camera_matrix, camera_trajectory

    def _get_colmap_processor(self, colmap_options: COLMAPOptions) -> COLMAPProcessor:
        processor = COLMAPProcessor(image_path=self.path_to_rgb_frames, workspace_path=self.path_to_colmap,
                                    colmap_options=colmap_options)

        return processor

    def _adjust_colmap_poses(self, colmap_options: COLMAPOptions) -> np.array:
        processor = self._get_colmap_processor(colmap_options)

        _, colmap_trajectory = processor.load_camera_params()

        t_cmp = colmap_trajectory[:, 4:].copy()
        t_cmp = t_cmp - t_cmp[0]
        # #TODO: Estimate appropriate scale and shift per dataset.
        # t_cmp /= 16

        t_cmp[:, 1], t_cmp[:, 2] = -t_cmp[:, 2].copy(), t_cmp[:, 1].copy()

        if colmap_options.use_raw_pose:
            t_cmp[:, 0] *= -1.

        R_cmp = colmap_trajectory[:, :4].copy()
        R_cmp[:, :3] = R_cmp[:, :3] - R_cmp[0, :3]

        R_cmp[:, 1], R_cmp[:, 2] = -R_cmp[:, 2].copy(), R_cmp[:, 1].copy()

        if colmap_options.use_raw_pose:
            R_cmp[:, 0] *= -1.

        refined_trajectory = np.eye(4)

        refined_trajectory[:, :4] = R_cmp
        refined_trajectory[:, 4:] = t_cmp

        return refined_trajectory

    def compare_gt_and_colmap_pose(self, colmap_options: COLMAPOptions):
        processor = COLMAPProcessor(image_path=self.path_to_rgb_frames, workspace_path=self.path_to_colmap,
                                    colmap_options=colmap_options)

        if not processor.probably_has_results:
            raise RuntimeError(f"Did not find COLMAP reconstruction data in the folder: {self.path_to_colmap}")

        _, colmap_trajectory = processor.load_camera_params()

        old_ct = np.loadtxt(self.path_to_camera_trajectory)
        t_gt = old_ct[:, 4:].copy()
        t_gt = t_gt - t_gt[0]

        t_cmp = colmap_trajectory[:, 4:].copy()
        t_cmp = t_cmp - t_cmp[0]
        t_cmp /= 16

        t_cmp[:, 1], t_cmp[:, 2] = -t_cmp[:, 2].copy(), t_cmp[:, 1].copy()

        if colmap_options.use_raw_pose:
            t_cmp[:, 0] *= -1.

        R_gt = Rotation.from_quat(old_ct[:, :4].copy()).as_euler('xyz', degrees=True)
        R_gt = R_gt - R_gt[0]

        R_cmp = colmap_trajectory[:, :4].copy()
        R_cmp[:, :3] = R_cmp[:, :3] - R_cmp[0, :3]

        R_cmp[:, 1], R_cmp[:, 2] = -R_cmp[:, 2].copy(), R_cmp[:, 1].copy()

        if colmap_options.use_raw_pose:
            R_cmp[:, 0] *= -1.

        colmap_trajectory[:, :4] = R_cmp
        colmap_trajectory[:, 4:] = t_cmp

        R_cmp = Rotation.from_quat(R_cmp).as_euler('xyz', degrees=True)

        # 2D Plots
        plt.close('all')
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))

        ax = axes[0, 0]
        ax.set_title('Trajectory (XY)')
        ax.plot(t_gt[:, 0], t_gt[:, 1], label='gt')
        ax.plot(t_cmp[:, 0], t_cmp[:, 1], label='cmp')
        ax.legend()

        ax = axes[0, 1]
        ax.set_title('Trajectory (XZ)')
        ax.plot(t_gt[:, 0], t_gt[:, 2], label='gt')
        ax.plot(t_cmp[:, 0], t_cmp[:, 2], label='cmp')
        ax.legend()

        ax = axes[1, 0]
        ax.set_title('Rotation (XY)')
        ax.plot(R_gt[:, 0], R_gt[:, 1], label='gt')
        ax.plot(R_cmp[:, 0], R_cmp[:, 1], label='cmp')
        ax.legend()

        ax = axes[1, 1]
        ax.set_title('Rotation (XZ)')
        ax.plot(R_gt[:, 0], R_gt[:, 2], label='gt')
        ax.plot(R_cmp[:, 0], R_cmp[:, 2], label='cmp')
        ax.legend()

        plt.tight_layout()
        plt.savefig('trajectory_viz.png')
        plt.close('all')

        # 3D Plot
        plt.close('all')
        fig = plt.figure(figsize=plt.figaspect(0.5))

        ax = fig.add_subplot(1, 2, 1, projection='3d')
        ax.set_title('Trajectory')
        ax.plot(t_gt[:, 0], t_gt[:, 1], t_gt[:, 2], label='gt')
        ax.plot(t_cmp[:, 0], t_cmp[:, 1], t_cmp[:, 2], label='cmp')
        ax.legend()

        ax = fig.add_subplot(1, 2, 2, projection='3d')
        ax.set_title('Rotation')
        ax.plot(R_gt[:, 0], R_gt[:, 1], R_gt[:, 2], label='gt')
        ax.plot(R_cmp[:, 0], R_cmp[:, 1], R_cmp[:, 2], label='cmp')
        ax.legend()

        plt.tight_layout()
        plt.savefig('trajectory_viz.png')
        plt.close('all')


class DatasetAdaptor(DatasetBase):
    """Creates a copy of a dataset in the VTMDataset format."""

    def __init__(self, base_path: File, output_path: File, overwrite_ok=False):
        """
        :param base_path: The path to the dataset.
        :param output_path: The path to write the new dataset to.
        :param overwrite_ok: Whether it is okay to overwrite existing adapted dataset.
        """
        super().__init__(base_path=base_path, overwrite_ok=overwrite_ok)

        self.output_path = output_path

    def convert(self) -> VTMDataset:
        """
        Read the dataset and create a copy in the VTMDataset format.

        :return: The newly created dataset object.
        """
        raise NotImplementedError


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

    def __init__(self, base_path: File, output_path: File, is_16_bit=True, overwrite_ok=False):
        """
        :param base_path: The path to folder containing the dataset.
        :param output_path: The path to write the new dataset to.
        :param is_16_bit: Whether the images are stored with 16-bit values or 32-bit values.
        :param overwrite_ok: Whether it is okay to overwrite existing depth maps and/or instance segmentation masks.
        """
        super().__init__(base_path=base_path, output_path=output_path, overwrite_ok=overwrite_ok)

        self.base_path = Path(base_path)
        self.pose_path = Path(os.path.join(base_path, str(Path(self.pose_path))))
        self.rgb_files_path = Path(os.path.join(base_path, str(Path(self.rgb_files_path))))
        self.depth_map_files_path = Path(os.path.join(base_path, str(Path(self.depth_map_files_path))))

        self.is_16_bit = is_16_bit
        # The depth maps need to be divided by 5000 for the 16-bit PNG files
        # or 1.0 (i.e. no effect) for the 32-bit float images in the ROS bag files
        self.depth_scale_factor = 1.0 / 5000.0 if is_16_bit else 1.0

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

    def convert(self) -> VTMDataset:
        log("Getting synced frame data...")
        image_filenames, depth_filenames, camera_trajectory = self._get_synced_frame_data()

        if VTMDataset.is_valid_folder_structure(self.output_path):
            log(f"Found cached dataset at {self.output_path}.")
            dataset = VTMDataset(self.output_path, overwrite_ok=self.overwrite_ok)

            num_frames = len(os.listdir(dataset.path_to_rgb_frames))
            num_depth_maps = len(os.listdir(dataset.path_to_depth_maps))
            # TODO: Check if length of camera trajectories match up.

            if (not num_frames == len(image_filenames)) or (not num_depth_maps == len(depth_filenames)):
                raise RuntimeError(f"Expected {len(image_filenames):03,d} frames, "
                                   f"found {num_frames:03,d} RGB frames and "
                                   f"{num_depth_maps:03,d} depth maps in {dataset.base_path}.")

            return dataset

        os.makedirs(self.output_path, exist_ok=self.overwrite_ok)

        join = os.path.join

        source_image_folder = join(self.base_path, self.rgb_folder)
        output_image_folder = join(self.output_path, VTMDataset.rgb_folder)
        os.makedirs(output_image_folder, exist_ok=self.overwrite_ok)

        source_depth_folder = join(self.base_path, self.depth_folder)
        output_depth_folder = join(self.output_path, VTMDataset.depth_folder)
        os.makedirs(output_depth_folder, exist_ok=self.overwrite_ok)

        log(f"Creating metadata for dataset.")
        # Depth scale here is it to 1 / 1000 because the depth maps will be converted to millimetres later on.
        metadata = DatasetMetadata(num_frames=len(image_filenames), fps=self.fps, width=self.width, height=self.height,
                                   depth_scale=1. / 1000.)
        metadata_path = os.path.join(self.output_path, VTMDataset.metadata_filename)
        metadata.save(metadata_path)

        log(f"Creating camera matrix file.")
        camera_matrix_path = os.path.join(self.output_path, VTMDataset.camera_matrix_filename)
        # noinspection PyTypeChecker
        np.savetxt(camera_matrix_path, self.intrinsic_matrix)

        log(f"Converting camera trajectory.")
        camera_trajectory_path = join(self.output_path, VTMDataset.camera_trajectory_filename)
        # noinspection PyTypeChecker
        np.savetxt(camera_trajectory_path, camera_trajectory)

        pool = ThreadPool(processes=psutil.cpu_count(logical=False))

        log(f"Copying RGB frames.")
        image_file_ext = Path(image_filenames[0]).suffix
        image_copy_jobs = [(join(source_image_folder, filename), join(output_image_folder, f"{i:06d}{image_file_ext}"))
                           for i, filename in enumerate(image_filenames)]
        pool.starmap(shutil.copy, image_copy_jobs)

        log(f"Copying depth maps.")

        def convert_depth_to_mm(input_path, output_path):
            depth_map = imageio.imread(input_path)
            depth_map = depth_map * self.depth_scale_factor  # convert to metres from non-standard scale & units.
            depth_map = (1000 * depth_map).astype(np.uint16)  # convert to 16-bit millimetres.
            imageio.imwrite(output_path, depth_map)

        depth_map_ext = Path(depth_filenames[0]).suffix
        depth_copy_jobs = [(join(source_depth_folder, filename), join(output_depth_folder, f"{i:06d}{depth_map_ext}"))
                           for i, filename in enumerate(depth_filenames)]
        pool.starmap(convert_depth_to_mm, depth_copy_jobs)

        log(f"Created new dataset at {self.output_path}.")

        return VTMDataset(self.output_path, overwrite_ok=self.overwrite_ok)


class Video2ImageFolder:
    """Collection of methods for converting an RGB video to a folder of frames."""

    @staticmethod
    def convert(video_path: File, output_path: File, output_ext='.png', overwrite_ok=False,
                transform: Callable[[np.ndarray], np.ndarray] = None):
        """
        Create a folder of images from an RGB video file.

        :param video_path: The path to the video.
        :param output_path: Where to save the images to.
        :param output_ext: The file extension of the output images.
        :param overwrite_ok: Whether it is okay to overwrite existing images in `output_path`.
        :param transform: (optional) A function that accepts a frame, applies some sequence of transform(s)
            and returns the new frame.
        :return: The path to the newly created image folder.
        """
        os.makedirs(output_path, exist_ok=overwrite_ok)

        frames = Video2ImageFolder._get_video_frames(video_path)

        for i, frame in enumerate(frames):
            filename = f"{i:06d}{output_ext}"
            frame_output_path = os.path.join(output_path, filename)

            if transform:
                frame = transform(frame)

            cv2.imwrite(frame_output_path, frame)

        return output_path

    @staticmethod
    def _get_video_frames(video_path):
        """
        Get frames from a video.

        :param video_path: The path to the video file.
        :return: Yields each frame (BGR, HWC format).
        """
        video = cv2.VideoCapture(video_path)

        if not video.isOpened():
            raise RuntimeError(f"Could not open RGB video file: {video_path}")

        num_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

        i = 0

        while video.grab():
            has_frame, frame = video.retrieve()
            if has_frame:
                yield frame

                i += 1
                log(f"[{i:03d}/{num_frames:03d}] Loading video...", prefix='\r', end='')
            else:
                break

        print()
        video.release()


class StrayScannerAdaptor(DatasetAdaptor):
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

    def __init__(self, base_path: Union[str, Path], output_path: Union[str, Path], overwrite_ok=False,
                 resize_to: Optional[Union[int, Size]] = None, depth_confidence_filter_level=0):
        """
        :param base_path: The path to the dataset.
        :param output_path: The path to write the new dataset to.
        :param overwrite_ok: Whether it is okay to overwrite existing depth maps and/or instance segmentation masks.
        :param resize_to: The resolution (width, height) to resize the images to.
        :param depth_confidence_filter_level: The minimum confidence value (0, 1, or 2) for the corresponding depth
                                              value to be kept. E.g. if set to 1, all pixels in the depth map where the
                                              corresponding pixel in the confidence map is less than 1 will be ignored.
        """
        super().__init__(base_path=base_path, output_path=output_path, overwrite_ok=overwrite_ok)

        self.camera_matrix: Optional[np.ndarray] = None
        self.camera_trajectory: Optional[np.ndarray] = None
        self.rgb_frames: Optional[np.ndarray] = None
        self.depth_maps: Optional[np.ndarray] = None
        self.masks: Optional[np.ndarray] = None

        if isinstance(resize_to, tuple):
            # Convert to HW from WH format.
            resize_to = tuple(reversed(resize_to))

        self.target_resolution = resize_to  # (height, width)
        self.depth_confidence_filter_level = depth_confidence_filter_level

        assert depth_confidence_filter_level in self.depth_confidence_levels, \
            f"Confidence filter must be one of the following: {self.depth_confidence_levels}."

    def convert(self) -> VTMDataset:
        video_metadata = self._get_video_metadata()

        if VTMDataset.is_valid_folder_structure(self.output_path):
            log(f"Found cached dataset at {self.output_path}.")
            dataset = VTMDataset(self.output_path, overwrite_ok=self.overwrite_ok)

            expected_num_frames = video_metadata.num_frames
            expected_num_depth_maps = len(os.listdir(os.path.join(self.base_path, self.depth_folder)))
            # expected_camera_trajectory_length = len(self._load_camera_trajectory())

            num_frames = len(os.listdir(dataset.path_to_rgb_frames))
            num_depth_maps = len(os.listdir(dataset.path_to_depth_maps))
            # TODO: Check if length of camera trajectories match up.
            # camera_trajectory_length = len(data.camera_trajectory)

            if (not num_frames == expected_num_frames) or (not num_depth_maps == expected_num_depth_maps):
                raise RuntimeError(f"Expected {expected_num_frames:03,d} frames, "
                                   f"found {num_frames:03,d} RGB frames and "
                                   f"{num_depth_maps:03,d} depth maps in {dataset.base_path}.")

            return dataset

        log(f"Creating new dataset at {self.output_path}.")
        os.makedirs(self.output_path, exist_ok=self.overwrite_ok)

        source_hw = video_metadata.height, video_metadata.width
        target_resolution = self._calculate_target_resolution(source_hw)
        log(f"Will resize frames from {video_metadata.width}x{video_metadata.height} to "
            f"{target_resolution[1]}x{target_resolution[0]} (width, height).")

        log(f"Creating metadata for dataset.")
        self._create_metadata(video_metadata, target_resolution)
        log(f"Creating camera matrix file.")
        self._convert_camera_matrix(source_hw, target_resolution)
        log(f"Converting camera trajectory.")
        self._convert_camera_trajectory()
        log(f"Converting video to folder of images.")
        self._convert_video(target_resolution)
        log(f"Converting depth maps to folder of image files.")
        self._convert_depth(target_resolution)

        log(f"Created new dataset at {self.output_path}.")

        return VTMDataset(self.output_path, overwrite_ok=self.overwrite_ok)

    def _calculate_target_resolution(self, source_hw):
        """
        Calculate the target resolution and perform some sanity checks.

        :param source_hw: The resolution of the input frames. These are used if the target resolution is given as a
            single value indicating the desired length of the longest side of the images.
        :return: The target resolution as a 2-tuple (height, width).
        """
        target_resolution = self.target_resolution

        if isinstance(target_resolution, int):
            # Cast results to int to avoid warning highlights in IDE.
            longest_side = int(np.argmax(source_hw))
            shortest_side = int(np.argmin(source_hw))

            new_size = [0, 0]
            new_size[longest_side] = target_resolution

            scale_factor = target_resolution / source_hw[longest_side]
            new_size[shortest_side] = int(source_hw[shortest_side] * scale_factor)

            target_resolution = new_size
        elif isinstance(target_resolution, tuple):
            # TODO: Change from assertions to raising exceptions.
            assert len(target_resolution) == 2, \
                f"The target resolution must be a 2-tuple, but got a {len(target_resolution)}-tuple."
            assert isinstance(target_resolution[0], int) and isinstance(target_resolution[1], int), \
                f"Expected target resolution to be a 2-tuple of integers, but got a tuple of" \
                f" ({type(target_resolution[0])}, {type(target_resolution[1])})."
        elif target_resolution is None:
            target_resolution = source_hw

        target_orientation = 'portrait' if np.argmax(target_resolution) == 0 else 'landscape'
        source_orientation = 'portrait' if np.argmax(source_hw) == 0 else 'landscape'

        if target_orientation != source_orientation:
            warnings.warn(
                f"The input images appear to be in {source_orientation} ({source_hw[1]}x{source_hw[0]}), "
                f"but they are being resized to what appears to be "
                f"{target_orientation} ({target_resolution[1]}x{target_resolution[0]})")

        return target_resolution

    def _get_video_metadata(self):
        """
        Get metadata about the source video.
        :return: A VideoMetadata object.
        """
        video_path = os.path.join(self.base_path, self.video_filename)
        video = cv2.VideoCapture(video_path)

        if not video.isOpened():
            raise RuntimeError(f"Could not open RGB video file: {video_path}")

        fps = float(video.get(cv2.CAP_PROP_FPS))
        width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        num_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

        video_metadata = VideoMetadata(video_path, width=width, height=height, num_frames=num_frames, fps=fps)

        return video_metadata

    def _create_metadata(self, video_metadata, target_resolution):
        """
        Create the dataset metadata file.

        :param video_metadata: The metadata of the input video file.
        :param target_resolution: The resolution that the final video frames will be resized to.
        """
        height, width = target_resolution
        # Input video is rotated 90 degrees anti-clockwise for some reason.
        # Need to swap height and width.
        width, height = height, width

        metadata = DatasetMetadata(num_frames=video_metadata.num_frames, fps=video_metadata.fps, width=width,
                                   height=height, depth_scale=1. / 1000.)
        metadata_path = os.path.join(self.output_path, VTMDataset.metadata_filename)
        metadata.save(metadata_path)

    def _convert_camera_trajectory(self):
        """
        Load the camera trajectory from disk, convert it to (r, t) format (where r is a quaternion) and save it in the
         dataset folder.
        """
        camera_trajectory = self._load_camera_trajectory()
        camera_trajectory_path = os.path.join(self.output_path, VTMDataset.camera_trajectory_filename)
        # noinspection PyTypeChecker
        np.savetxt(camera_trajectory_path, camera_trajectory)

    def _convert_camera_matrix(self, source_hw, target_resolution):
        """
        Load the camera matrix, adjust it for the new target resolution and save it in the new dataset folder.

        :param source_hw: The height and width of the original video frames.
        :param target_resolution: The resolution that the final video frames will be resized to.
        """
        camera_matrix = self._load_camera_matrix(scale_x=target_resolution[1] / source_hw[1],
                                                 scale_y=target_resolution[0] / source_hw[0])
        camera_matrix_path = os.path.join(self.output_path, VTMDataset.camera_matrix_filename)
        # noinspection PyTypeChecker
        np.savetxt(camera_matrix_path, camera_matrix)

    def _load_camera_matrix(self, scale_x=1.0, scale_y=1.0):
        """
        Load the camera intrinsic parameters.

        :param scale_x: The scale factor to apply to the x component of the focal length and principal point.
        :param scale_y: The scale factor to apply to the y component of the focal length and principal point.
        :return: The 3x3 camera matrix.
        """
        intrinsics_path = os.path.join(self.base_path, self.camera_matrix_filename)
        camera_matrix = np.loadtxt(intrinsics_path, delimiter=',')

        camera_matrix[0, 0] *= scale_x
        camera_matrix[0, 2] *= scale_x
        camera_matrix[1, 1] *= scale_y
        camera_matrix[1, 2] *= scale_y

        # Input video is rotated 90 degrees anti-clockwise for some reason.
        # Need to adjust camera matrix and RGB-D data so that it is the right way up.
        camera_matrix[0, 0], camera_matrix[1, 1] = camera_matrix[1, 1], camera_matrix[0, 0]
        camera_matrix[0, 2], camera_matrix[1, 2] = camera_matrix[1, 2], camera_matrix[0, 2]

        return camera_matrix

    def _load_camera_trajectory(self):
        """
        Load the camera poses.

        :return: The Nx6 matrix where each row contains the rotation in axis-angle format and the translation vector.
        """
        # Code adapted from https://github.com/kekeblom/StrayVisualizer/blob/df5f39c750e8eec62b130dc9c8a91bdbcff1d952/stray_visualize.py#L43
        trajectory_path = os.path.join(self.base_path, self.camera_trajectory_filename)
        # The first row is the header row, so skip
        trajectory_raw = np.loadtxt(trajectory_path, delimiter=',', skiprows=1)

        trajectory = []

        for line in trajectory_raw:
            # timestamp, frame, ...
            _, _, tx, ty, tz, qx, qy, qz, qw = line
            trajectory.append((qx, qy, qz, qw, tx, ty, tz))

        trajectory = np.ascontiguousarray(trajectory)

        return trajectory

    def _convert_video(self, target_resolution):
        """
        Convert the input video to a folder of video frames.

        :param target_resolution: The resolution that the final video frames will be resized to.
        """
        # OpenCV's resize takes in a tuple in WH format (width, height), however `target_resolution` is in HW.
        target_resolution_cv = tuple(reversed(target_resolution))

        def image_transform(frame):
            frame = cv2.resize(frame, target_resolution_cv)
            # Input video is rotated 90 degrees anti-clockwise for some reason.
            # Need to adjust camera matrix and RGB-D data so that it is the right way up.
            frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)

            return frame

        video_path = os.path.join(self.base_path, self.video_filename)
        output_images_path = os.path.join(self.output_path, VTMDataset.rgb_folder)
        Video2ImageFolder.convert(video_path, output_images_path, overwrite_ok=self.overwrite_ok,
                                  transform=image_transform)

    def _convert_depth(self, target_resolution):
        """
        Convert the depth maps from .npy (NumPy) to .png files.

        :param target_resolution: The resolution that the final video frames will be resized to.
        """
        filter_level = self.depth_confidence_filter_level
        # OpenCV's resize takes in a tuple in WH format (width, height), however `target_resolution` is in HW.
        target_resolution_cv = tuple(reversed(target_resolution))

        source_depth_path = os.path.join(self.base_path, self.depth_folder)
        output_depth_path = os.path.join(self.output_path, VTMDataset.depth_folder)
        confidence_path = os.path.join(self.base_path, self.confidence_map_folder)

        os.makedirs(output_depth_path, exist_ok=self.overwrite_ok)

        def convert_depth_map(i, filename):
            depth_map_path = os.path.join(source_depth_path, filename)
            depth_map = np.load(depth_map_path)

            if depth_map.dtype != np.uint16:
                raise RuntimeError(f"Expected 16-bit depth maps, got {depth_map.dtype}.")

            confidence_map_filename = f"{Path(filename).stem}.png"
            confidence_map_path = os.path.join(confidence_path, confidence_map_filename)
            confidence_map = cv2.imread(confidence_map_path, cv2.IMREAD_GRAYSCALE)

            depth_map[confidence_map < filter_level] = 0

            depth_map = cv2.resize(depth_map, target_resolution_cv)
            # Input video is rotated 90 degrees anti-clockwise for some reason.
            # Need to adjust camera matrix and RGB-D data so that it is the right way up.
            depth_map = cv2.rotate(depth_map, cv2.ROTATE_90_CLOCKWISE)

            output_depth_map_path = os.path.join(output_depth_path, f"{i:06d}.png")
            cv2.imwrite(output_depth_map_path, depth_map)

        pool = ThreadPool(psutil.cpu_count(logical=False))
        pool.starmap(convert_depth_map, enumerate(sorted(os.listdir(source_depth_path))))


class UnrealDatasetInfo:
    def __init__(self, width, height, num_frames, fps=30.0, max_depth=10.0, invalid_depth_value=0.0,
                 is_16bit_depth=True, intrinsics_filename='camera.txt', trajectory_filename='trajectory.txt',
                 colour_folder='colour', depth_folder='depth'):
        """
        :param width: The width in pixels of the colour frames and depth maps.
        :param height: The height in pixels of the colour frames and depth maps.
        :param num_frames: The number of frames in the dataset.
        :param fps: The framerate of the captured video.
        :param max_depth: The maximum depth value allowed in a depth map.
        :param invalid_depth_value: The values used to indicate invalid (e.g. missing) depth.
        :param is_16bit_depth: Whether the depth maps use 16-bit values.
        :param intrinsics_filename: The name of camera parameters file.
        :param trajectory_filename: The name of the camera pose file.
        :param colour_folder: The name of the folder that contains the colour frames.
        :param depth_folder: The name of the folder that contains the depth maps.
        """
        self.width = width
        self.height = height
        self.num_frames = num_frames
        self.fps = fps
        self.max_depth = max_depth
        self.invalid_depth_value = invalid_depth_value
        self.is_16bit_depth = is_16bit_depth
        self.intrinsics_filename = intrinsics_filename
        self.trajectory_filename = trajectory_filename
        self.colour_folder = colour_folder
        self.depth_folder = depth_folder

    def save_json(self, fp):
        if isinstance(fp, str):
            with open(fp, 'w') as f:
                json.dump(self.__dict__, f)
        else:
            json.dump(self.__dict__, fp)

    @staticmethod
    def from_json(fp):
        if isinstance(fp, str):
            with open(fp, 'r') as f:
                data = json.load(f)
        else:
            data = json.load(fp)

        return UnrealDatasetInfo(**data)


class UnrealAdaptor(DatasetAdaptor):
    """Convert datasets created in Unreal Engine (https://github.com/eight0153/UnrealDataset.git) to the VTMDataset format."""

    metadata_filename = 'info.json'
    camera_matrix_filename = 'camera.txt'
    camera_trajectory_filename = 'trajectory.txt'
    required_files = [camera_matrix_filename, camera_trajectory_filename]

    rgb_folder = 'colour'
    depth_folder = 'depth'
    required_folders = [rgb_folder, depth_folder]

    def convert(self) -> VTMDataset:
        join = os.path.join

        image_filenames = os.listdir(join(self.base_path, self.rgb_folder))
        depth_filenames = os.listdir(join(self.base_path, self.depth_folder))

        dataset_metadata = UnrealDatasetInfo.from_json(join(self.base_path, self.metadata_filename))
        camera_matrix = np.loadtxt(join(self.base_path, self.camera_matrix_filename))
        camera_trajectory = np.loadtxt(join(self.base_path, self.camera_trajectory_filename))

        if VTMDataset.is_valid_folder_structure(self.output_path):
            log(f"Found cached dataset at {self.output_path}.")
            dataset = VTMDataset(self.output_path, overwrite_ok=self.overwrite_ok)

            num_frames = len(os.listdir(dataset.path_to_rgb_frames))
            num_depth_maps = len(os.listdir(dataset.path_to_depth_maps))
            trajectory_length = len(dataset.camera_trajectory)

            if (not num_frames == len(image_filenames)) or (not num_depth_maps == len(depth_filenames)):
                raise RuntimeError(f"Expected {len(image_filenames):03,d} frames, "
                                   f"found {num_frames:03,d} RGB frames and "
                                   f"{num_depth_maps:03,d} depth maps in {dataset.base_path}.")

            if trajectory_length != len(camera_trajectory):
                raise RuntimeError(f"Expected a camera trajectory with {len(camera_trajectory):03,d} poses, "
                                   f"found {trajectory_length:03,d} poses {dataset.base_path}.")

            return dataset

        os.makedirs(self.output_path, exist_ok=self.overwrite_ok)

        log(f"Creating metadata for dataset.")

        metadata = DatasetMetadata(num_frames=dataset_metadata.num_frames, fps=dataset_metadata.fps,
                                   width=dataset_metadata.width, height=dataset_metadata.height, depth_scale=1. / 1000.)
        metadata.save(join(self.output_path, VTMDataset.metadata_filename))

        log(f"Creating camera matrix file.")

        # noinspection PyTypeChecker
        np.savetxt(join(self.output_path, VTMDataset.camera_matrix_filename), camera_matrix)

        log(f"Converting camera trajectory.")
        converted_trajectory = []

        for (rx, ry, rz, tx, ty, tz) in camera_trajectory:
            qx, qy, qz, qw = Rotation.from_rotvec((rx, ry, rz)).as_quat()

            converted_trajectory.append((qx, qy, qz, qw, tx, ty, tz))

        converted_trajectory = np.array(converted_trajectory)

        # noinspection PyTypeChecker
        np.savetxt(join(self.output_path, VTMDataset.camera_trajectory_filename), converted_trajectory)

        log(f"Copying RGB frames to new dataset.")

        pool = ThreadPool(processes=psutil.cpu_count(logical=False))

        def copy_frames(source_path, dest_path, files, copy_func=shutil.copy):
            os.makedirs(dest_path, exist_ok=self.overwrite_ok)

            extension = Path(files[0]).suffix
            copy_jobs = [(join(source_path, filename), join(dest_path, f"{i:06d}{extension}"))
                         for i, filename in enumerate(files)]

            pool.starmap(copy_func, copy_jobs)

        image_source_path = join(self.base_path, self.rgb_folder)
        image_dest_path = join(self.output_path, VTMDataset.rgb_folder)
        copy_frames(image_source_path, image_dest_path, image_filenames)

        log(f"Copying depth maps to new dataset.")

        depth_map_max_value = np.iinfo(np.uint16).max if dataset_metadata.is_16bit_depth else np.iinfo(np.uint8).max
        depth_scale = dataset_metadata.max_depth / depth_map_max_value

        def convert_depth_to_mm(input_path, output_path):
            depth_map = imageio.imread(input_path)
            depth_map = depth_scale * depth_map
            depth_map = (1000 * depth_map).astype(np.uint16)
            imageio.imwrite(output_path, depth_map)

        depth_source_path = join(self.base_path, self.depth_folder)
        depth_dest_path = join(self.output_path, VTMDataset.depth_folder)
        copy_frames(depth_source_path, depth_dest_path, depth_filenames, copy_func=convert_depth_to_mm)

        log(f"Created new dataset at {self.output_path}.")

        dataset = VTMDataset(self.output_path, overwrite_ok=self.overwrite_ok)

        return dataset


class BundleFusionConfig:
    def __init__(self, **kwargs):
        self.config_dict = OrderedDict(**kwargs)

    def __getitem__(self, key):
        return self.config_dict[key]

    def __setitem__(self, key, value):
        if key in self.config_dict and (value_type := type(value)) != (expected_type := type(self.config_dict[key])):
            warnings.warn(f"The config file entry \"{key}\" is of type {expected_type} "
                          f"but it is being set to a new value of type {value_type}")

        self.config_dict[key] = value

    @staticmethod
    def load(f) -> 'BundleFusionConfig':
        if isinstance(f, str):
            with open(f, 'r') as fp:
                return BundleFusionConfig._read_file(fp)
        else:
            return BundleFusionConfig._read_file(f)

    @staticmethod
    def _read_file(fp) -> 'BundleFusionConfig':
        config = OrderedDict()

        delimiter_pattern = re.compile("[;#]|(//)")

        def convert_value(string_value):
            if string_value[0] == '"' and string_value[-1] == '"':
                return string_value.strip('"')
            elif string_value == 'true':
                return True
            elif string_value == 'false':
                return False
            elif string_value[-1] == 'f':
                return float(string_value[:-1])
            else:
                return int(string_value)

        for line in fp:
            line = line.strip()

            if delimiter_match := re.search(delimiter_pattern, line):
                line = line[:delimiter_match.start()]

            if len(line) < 1:
                continue

            attribute_name, values = line.split("=")
            attribute_name = attribute_name.strip()
            values = values.strip()

            if len(attribute_name) < 1 or len(values) < 1:
                continue

            parts = values.split(" ")

            if len(parts) > 1:
                converted_values = [convert_value(value) for value in parts]
            else:
                converted_values = convert_value(values)

            config[attribute_name] = converted_values

        return BundleFusionConfig(**config)

    def save(self, f):
        if isinstance(f, str):
            with open(f, 'w') as fp:
                self._write_to_disk(fp)
        else:
            self._write_to_disk(f)

    def _write_to_disk(self, fp):
        def convert_to_string(value):
            if type(value) == list:
                return ' '.join([convert_to_string(item) for item in value])
            elif type(value) == float:
                return f"{value}f"
            elif type(value) == int:
                return str(value)
            elif type(value) == str:
                return f"\"{value}\""
            elif type(value) == bool:
                return str(value).lower()
            else:
                raise ValueError(
                    f"The type '{type(value)}' is not supported for serialisation. Supported types are list, float, int and str.")

        for attribute_name, value in self.config_dict.items():
            line = f"{attribute_name} = {convert_to_string(value)};\n"
            fp.write(line)
