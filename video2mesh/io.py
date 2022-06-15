import datetime
import json
import os
import shutil
import struct
import subprocess
import warnings
from os.path import join as pjoin
from pathlib import Path
from typing import Union, Tuple, Optional, Callable, IO

import cv2
import imageio
import numpy as np
import torch
from PIL import Image
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.engine import DefaultPredictor
from scipy.spatial.transform import Rotation
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from video2mesh.geometry import pose_vec2mat, normalise_trajectory
from video2mesh.image_processing import dilate_mask
from video2mesh.options import COLMAPOptions, MaskDilationOptions
from video2mesh.utils import log, tqdm_imap
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
                 overwrite_ok=False, for_colmap=False, filename_fmt: Optional[Callable[[int], str]] = None):
    """
    Create instance segmentation masks for the given RGB video sequence and save the masks to disk.

    :param rgb_loader: The PyTorch DataLoader that loads the RGB frames (no data augmentations applied).
    :param mask_folder: The path to save the masks to.
    :param overwrite_ok: Whether it is okay to write over any mask files in `mask_folder` if it already exists.
    :param for_colmap: Whether the masks are intended for use with COLMAP or 3D video generation.
        Masks will be black and white with the background coloured white and using the
        corresponding input image's filename.
    :param filename_fmt: (optional) a function that generates a frame filename from the frame index,
        e.g. 123 -> '000123.png'.
    """
    log(f"Creating masks...")

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
    log(class_names)

    person_label = class_names.index('person')
    predictor = BatchPredictor(cfg)

    log(f"Creating segmentation masks...")
    os.makedirs(mask_folder, exist_ok=overwrite_ok)
    i = 0

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
    def result_path(self):
        return pjoin(self.workspace_path, 'sparse')

    @property
    def probably_has_results(self):
        recon_result_path = pjoin(self.result_path, '0')
        min_files_for_recon = 4

        return os.path.isdir(self.result_path) and len(os.listdir(self.result_path)) > 0 and \
               (os.path.isdir(recon_result_path) and len(os.listdir(recon_result_path)) >= min_files_for_recon)

    def run(self):
        os.makedirs(self.workspace_path, exist_ok=True)

        if not os.path.isdir(self.mask_path) or len(os.listdir(self.mask_path)) == 0:
            log(f"Could not find masks in folder: {self.mask_path}.")
            log(f"Creating masks for COLMAP...")
            rgb_loader = DataLoader(ImageFolderDataset(self.image_path), batch_size=8, shuffle=False)
            create_masks(rgb_loader, self.mask_path, overwrite_ok=True, for_colmap=True)
        else:
            log(f"Found {len(os.listdir(self.mask_path))} masks in {self.mask_path}.")

        command = self.get_command()

        # TODO: Check that COLMAP is using GPU
        with subprocess.Popen(command, stdout=subprocess.PIPE, bufsize=1, universal_newlines=True) as p:
            for line in p.stdout:
                log(line, end='')

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

        models = sorted(os.listdir(self.result_path))
        num_models = len(models)

        if num_models == 1:
            sparse_recon_path = pjoin(self.result_path, models[0])
        else:
            log(f"COLMAP reconstructed {num_models} models when 1 was expected. Attempting to merge models....")

            path_to_merged = pjoin(self.result_path, 'merged')
            os.mkdir(path_to_merged)

            input_pairs = [(models[0], models[1])]

            if num_models > 2:
                for model in models[2:]:
                    input_pairs.append((path_to_merged, model))

            merge_successful = True

            for input1, input2 in input_pairs:
                # Use temporary folder for output to avoid any issues with reading/writing to same folder for num_models > 2.
                tmp_merged_folder = pjoin(self.result_path, 'tmp')

                if os.path.isdir(tmp_merged_folder):
                    shutil.rmtree(tmp_merged_folder)

                os.mkdir(tmp_merged_folder)

                merge_command = ['colmap', 'model_merger',
                                 '--input_path1', pjoin(self.result_path, input1),
                                 '--input_path2', pjoin(self.result_path, input2),
                                 f"--output_path", tmp_merged_folder]

                with subprocess.Popen(merge_command, stdout=subprocess.PIPE, bufsize=1, universal_newlines=True) as p:
                    for line in p.stdout:
                        log(line, end='')

                        if 'merge failed' in line.lower():
                            merge_successful = False

                p.wait()
                shutil.move(tmp_merged_folder, path_to_merged)

            if p.returncode == 0 and merge_successful:
                log(f"Merged {num_models} successfully, refining merged data with bundle adjustment...")

                path_to_refined = pjoin(self.result_path, 'merged_refined')
                os.mkdir(path_to_refined)

                bundle_adjustment_command = ['colmap', 'bundle_adjuster',
                                             '--input_path', path_to_merged,
                                             '--output_path', path_to_refined]

                with subprocess.Popen(bundle_adjustment_command, stdout=subprocess.PIPE, bufsize=1,
                                      universal_newlines=True) as p:
                    for line in p.stdout:
                        log(line, end='')

                p.wait()

                sparse_recon_path = path_to_refined
            else:
                log(f"Did not merge the {num_models} sub-models successfully, skipping bundle adjustment.")
                sparse_recon_path = path_to_merged

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
        log("Read intrinsic parameters.")

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
        extrinsic = self._adjust_colmap_poses(extrinsic, use_raw_pose=raw_pose)

        log(f"Read extrinsic parameters for {len(extrinsic)} frames.")

        return intrinsic, extrinsic

    @staticmethod
    def _adjust_colmap_poses(camera_trajectory: np.ndarray, use_raw_pose=False) -> np.array:
        t_cmp = camera_trajectory[:, 4:].copy()
        t_cmp = t_cmp - t_cmp[0]

        t_cmp[:, 1], t_cmp[:, 2] = -t_cmp[:, 2].copy(), t_cmp[:, 1].copy()

        if use_raw_pose:
            t_cmp[:, 0] *= -1.

        R_cmp = camera_trajectory[:, :4].copy()
        R_cmp[:, :3] = R_cmp[:, :3] - R_cmp[0, :3]

        R_cmp[:, 1], R_cmp[:, 2] = -R_cmp[:, 2].copy(), R_cmp[:, 1].copy()

        if use_raw_pose:
            R_cmp[:, 0] *= -1.

        refined_trajectory = np.zeros(shape=(len(R_cmp), R_cmp.shape[1] + t_cmp.shape[1]))

        refined_trajectory[:, :4] = R_cmp
        refined_trajectory[:, 4:] = t_cmp

        return refined_trajectory

    def get_sparse_depth_maps(self):
        cameras, images, points3d = self._load_model()

        camera_matrix, camera_trajectory = self.load_camera_params()

        source_image_shape = cv2.imread(pjoin(self.image_path, images[1].name)).shape[:2]
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


class ImageFolderDataset(Dataset):
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

    def __init__(self, num_frames: int, fps: float, width: int, height: int, is_gt: bool,
                 depth_mask_dilation_iterations: int, depth_scale: float, max_depth=10.0, frame_step=1):
        """
        :param num_frames: The number of frames in the video sequence.
        :param fps: The framerate at which the video was captured.
        :param width: The width of a frame (pixels).
        :param height: The height of a frame (pixels).
        :param is_gt: Whether the dataset was created using ground truth data for the camera parameters and depth maps.
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
        self.is_gt = is_gt

        if not isinstance(is_gt, bool):
            raise ValueError(f"is_gt must be a boolean, got {type(is_gt)}.")

        if not isinstance(num_frames, int) or num_frames < 1:
            raise ValueError(f"num_frames must be a positive integer, got {num_frames}.")

        if not isinstance(frame_step, int) or frame_step < 1:
            raise ValueError(f"frame_step must be a positive integer, got {frame_step}.")

        if not isinstance(width, int) or width < 1:
            raise ValueError(f"width must be a positive integer, got {width}.")

        if not isinstance(height, int) or height < 1:
            raise ValueError(f"height must be a positive integer, got {height}.")

        if not isinstance(depth_scale, float):
            raise ValueError(f"depth_scale must be a real number (float), got {type(depth_scale)}.")

        if not isinstance(max_depth, float) or max_depth < 0.0:
            raise ValueError(f"max_depth must be a positive, real number (float), got {max_depth} ({type(max_depth)}).")

        if not isinstance(depth_mask_dilation_iterations, int) or depth_mask_dilation_iterations < 1:
            raise ValueError(f"depth_mask_dilation_iterations must be a positive integer, "
                             f"got {depth_mask_dilation_iterations}.")

    def __eq__(self, other: 'DatasetMetadata') -> bool:
        return self.num_frames == other.num_frames and \
               np.isclose(self.fps, other.fps) and \
               self.frame_step == other.frame_step and \
               self.width == other.width and \
               self.height == other.height and \
               np.isclose(self.depth_scale, other.depth_scale) and \
               np.isclose(self.max_depth, other.max_depth) and \
               self.depth_mask_dilation_iterations == other.depth_mask_dilation_iterations and \
               self.is_gt == other.is_gt

    def __repr__(self):
        return f"{self.__class__.__name__}(num_frames={self.num_frames}, fps={self.fps}, " \
               f"frame_step={self.frame_step}, width={self.width}, height={self.height}, " \
               f"max_depth={self.max_depth}, is_gt={self.is_gt}, " \
               f"depth_mask_dilation_iterations={self.depth_mask_dilation_iterations}, depth_scale={self.depth_scale})"

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

        return DatasetMetadata(num_frames=int(kwargs['num_frames']),
                               frame_step=int(kwargs['frame_step']),
                               fps=float(kwargs['fps']),
                               width=int(kwargs['width']),
                               height=int(kwargs['height']),
                               is_gt=bool(kwargs['is_gt']),
                               depth_scale=float(kwargs['depth_scale']),
                               max_depth=float(kwargs['max_depth']),
                               depth_mask_dilation_iterations=int(kwargs['depth_mask_dilation_iterations']))


class VTMDataset(DatasetBase):
    """The main dataset format for the video2mesh (VTM) project."""

    metadata_filename = "metadata.json"
    camera_matrix_filename = "camera_matrix.txt"
    camera_trajectory_filename = "camera_trajectory.txt"

    required_files = [metadata_filename, camera_trajectory_filename, camera_matrix_filename]

    rgb_folder = "rgb"
    depth_folder = "depth"
    mask_folder = "mask"
    masked_depth_folder = 'masked_depth'

    required_folders = [rgb_folder, depth_folder]

    # Dataset adaptors are expected to convert depth maps to mm.
    # This scaling factor converts the mm depth values to meters.
    depth_scaling_factor = 1. / 1000.

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
        self.mask_dataset = ImageFolderDataset(self.path_to_masks)

        self._masked_depth_path: Optional[str] = None

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

    def __len__(self):
        return self.num_frames

    def _get_depth_map_transform(self):
        def transform(depth_map):
            depth_map = self.depth_scaling_factor * depth_map.astype(np.float32)
            depth_map[depth_map > self.metadata.max_depth] = 0.0

            return depth_map

        return transform

    def create_masked_depth(self, dilation_options=MaskDilationOptions(num_iterations=64)) -> 'VTMDataset':
        start = datetime.datetime.now()

        masked_depth_folder = self.masked_depth_folder
        masked_depth_path = pjoin(self.base_path, masked_depth_folder)

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

        log(f"Writing masked depth to {masked_depth_path}...")
        args = list(zip(range(len(self)), self.depth_dataset, self.mask_dataset))
        tqdm_imap(save_depth_wrapper, args)

        self.metadata.depth_mask_dilation_iterations = dilation_options.num_iterations
        self.metadata.save(self.path_to_metadata)
        log(f"Update metadata")

        elapsed = datetime.datetime.now() - start

        log(f"Created {len(os.listdir(masked_depth_path))} masked depth maps in {elapsed}")

        return self

    def _load_camera_parameters(self, normalise=True) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load the ground truth camera matrix and trajectory from disk.

        The camera matrix is expected to be saved as a 3x3 matrix in a file called "camera.txt".
        The camera trajectory is expected to be saved as a Nx7 matrix in a file called "trajectory.txt", where N is the
        number of frames in the sequence and each row is a quaternion rotation 'r' and translation vector 't'.

        :param normalise: Whether to normalise the pose data s.t. the first pose is the identity.
        """
        camera_matrix = np.loadtxt(self.path_to_camera_matrix, dtype=np.float32)
        camera_trajectory = np.loadtxt(self.path_to_camera_trajectory, dtype=np.float32)

        if camera_matrix.shape != (3, 3):
            raise RuntimeError(f"Expected camera matrix to be a 3x3 matrix,"
                               f" but got {camera_matrix.shape} instead.")

        if len(camera_trajectory.shape) != 2 or camera_trajectory.shape[1] != 7:
            raise RuntimeError(f"Expected camera trajectory to be a Nx7 matrix,"
                               f" but got {camera_trajectory.shape} instead.")

        if normalise:
            camera_trajectory = normalise_trajectory(camera_trajectory)

        return camera_matrix, camera_trajectory

    @staticmethod
    def index_to_filename(index: int, file_extension="png") -> str:
        return f"{index:06d}.{file_extension}"