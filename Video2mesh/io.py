import datetime
import json
import os
import struct
import subprocess
import time
import warnings
from hashlib import sha256
from multiprocessing.pool import ThreadPool
from pathlib import Path
from typing import Union, List, Tuple, Optional, Callable, IO

import cv2
import numpy as np
import psutil
import torch
from PIL import Image
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.engine import DefaultPredictor
from scipy.spatial.transform import Rotation
from torch.utils.data import DataLoader, Dataset

from Video2mesh.options import COLMAPOptions, StorageOptions
from Video2mesh.utils import Timer, log
from thirdparty.AdaBins.infer import InferenceHelper
from thirdparty.colmap.scripts.python.read_write_model import read_model


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


def load_camera_parameters(storage_options, timer=Timer()):
    """
    Load
    :param storage_options:
    :param timer:
    :return:
    """
    # TODO: Complete docstring for this method.
    camera_params = np.loadtxt(os.path.join(storage_options.base_path, "camera.txt"))
    camera_trajectory = np.loadtxt(os.path.join(storage_options.base_path, "trajectory.txt"))

    if camera_params.shape != (3, 3):
        raise RuntimeError(f"Expected camera parameters (intrinsic) to be a (3, 3) matrix,"
                           f" but got {camera_params.shape} instead.")

    if camera_trajectory.shape[1] != 6:
        raise RuntimeError(f"Expected camera trajectory to be a (N, 6) matrix,"
                           f" but got {camera_trajectory.shape} instead.")

    timer.split("load camera parameters")
    return camera_params, camera_trajectory


def load_input_data(storage_options, depth_options, batch_size=-1, should_create_masks=False, timer=Timer()):
    """
    Load RGB-D frames and instance segmentation masks from disk.

    :param storage_options: The data about where to save outputs to.
    :param depth_options: The data about how depth maps should be loaded.
    :param batch_size: How many files to load at once when creating masks. Has little impact on performance.
    :param should_create_masks: Whether instance segmentation masks should be created if they do not already exist.
    :param timer: Timer object for simple benchmarking of code segments.

    :return: A 3-tuple of the RGB frames (N, H, W, C), depth maps (N, H, W) and masks (N, H, W).
    """
    # TODO: Set sensible default for batch size if arg value is -1.
    storage = storage_options

    if os.path.isdir(storage.mask_path) and \
            len(os.listdir(storage.mask_path)) == len(os.listdir(storage.colour_path)):
        print(f"Found cached masks in {storage.mask_path}")
    elif should_create_masks:
        rgb_loader = DataLoader(ImageFolderDataset(storage.colour_path),
                                batch_size=batch_size, shuffle=False)
        create_masks(rgb_loader, storage.mask_path, overwrite_ok=storage.overwrite_ok)
    else:
        raise RuntimeError(f"Masks not found in path {storage.mask_path} or number of masks do not match the "
                           f"number of rgb frames in {storage.colour_path}.")

    timer.split('locate/create masks')

    pool = ThreadPool(psutil.cpu_count(logical=False))

    rgb_frames = pool.map(cv2.imread, ImageFolderDataset(storage.colour_path).image_paths)

    depth_maps = pool.map(lambda path: cv2.imread(path, cv2.IMREAD_ANYDEPTH),
                          ImageFolderDataset(storage.depth_path).image_paths)

    masks = pool.map(lambda path: cv2.imread(path, cv2.IMREAD_GRAYSCALE),
                     ImageFolderDataset(storage.mask_path).image_paths)

    timer.split('load frame data to RAM')

    rgb_frames = np.asarray(rgb_frames)
    depth_maps = np.asarray(depth_maps)
    masks = np.asarray(masks)
    timer.split('concatenate frame data into NumPy array')

    rgb_frames = rgb_frames[:, :, :, ::-1]
    timer.split('convert frames from BGR to RGB')

    rgb_frames = np.flip(rgb_frames, axis=1)
    depth_maps = np.flip(depth_maps, axis=1)
    masks = np.flip(masks, axis=1)
    timer.split('put frame data the right way up')

    assert depth_maps.dtype == depth_options.depth_dtype, \
        f"Expected depth maps to be {depth_options.depth_dtype}, but got {depth_maps.dtype} instead."

    depth_maps = depth_maps / np.iinfo(depth_options.depth_dtype).max
    depth_maps *= depth_options.max_depth
    timer.split('scale depth')

    if rgb_frames.shape[:3] != depth_maps.shape != masks.shape:
        raise RuntimeError(f"RGB frames, depth maps and masks should have the same shape, "
                           f"but got {rgb_frames.shape}, {depth_maps.shape} and {masks.shape}")

    return rgb_frames, depth_maps, masks


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
                output_filename = f"{i:03d}.png"

            Image.fromarray(combined_masks).convert('L').save(os.path.join(mask_folder, output_filename))

            i += 1

        print(f"\r{i:03,d}/{len(rgb_loader.dataset):03,d}", end='')

    print()


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

            if image.mode != 'L':
                image = image.convert('RGB')

            image = np.asarray(image)

        if self.transform:
            image = self.transform(image)

        return image

    def __len__(self):
        return len(self.image_paths)


def numpy_to_ply(vertex, color=None, normals=None):
    n = vertex.shape[0]
    ply_data = []

    if color is None:
        color = 255 * np.ones((n, 3), dtype=np.int8)
    add_normals = True
    if normals is None:
        add_normals = False

    for i in range(n):
        data = {'x': vertex[i, 0], 'y': vertex[i, 1], 'z': vertex[i, 2],
                'red': color[i, 0], 'green': color[i, 1], 'blue': color[i, 2]}

        if add_normals:
            data['nx'], data['ny'], data['nz'] = normals[i, :]

        ply_data.append(data)

    return ply_data


def write_ply(full_name, vertex_data, face_data=None, meshcolor=0, face_uv=None, face_colors=None,
              texture_name='Parameterization.jpg'):
    write_normals = False
    if 'nx' in list(vertex_data[0].keys()):
        write_normals = True

    write_facedata = False
    if face_data is not None:
        write_facedata = True

    fid = open(full_name, 'w')
    fid.write('ply\n')
    fid.write('format ascii 1.0\n')
    if meshcolor == 1:
        fid.write('comment TextureFile %s\n' % texture_name)
    fid.write('element vertex %d\n' % len(vertex_data))
    fid.write('property float x\n')
    fid.write('property float y\n')
    fid.write('property float z\n')
    if meshcolor == 0:
        fid.write('property uchar red\n')
        fid.write('property uchar green\n')
        fid.write('property uchar blue\n')
    if write_normals:
        fid.write('property float nx\n')
        fid.write('property float ny\n')
        fid.write('property float nz\n')

    if write_facedata:
        fid.write('element face %d\n' % len(face_data))
        fid.write('property list uchar int vertex_indices\n')
        if meshcolor == 1:
            fid.write('property list uint8 float texcoord\n')
        elif meshcolor == 2:
            fid.write('property uchar red\n')
            fid.write('property uchar green\n')
            fid.write('property uchar blue\n')
    fid.write('end_header\n')

    for i in range(len(vertex_data)):
        fid.write('%.5f %.5f %.5f' % (vertex_data[i]['x'], vertex_data[i]['y'], vertex_data[i]['z']))
        if meshcolor == 0:
            fid.write(' %d %d %d\n' % (vertex_data[i]['red'], vertex_data[i]['green'], vertex_data[i]['blue']))
        else:
            fid.write('\n')

    if write_facedata:
        for i in range(len(face_data)):
            fid.write('3 %d %d %d\n' % (face_data[i][0], face_data[i][1], face_data[i][2]))
            if meshcolor == 1:
                fid.write('6 %.5f %.5f %.5f %.5f %.5f %.5f\n' % (
                    face_uv[i][0, 0], face_uv[i][1, 0], face_uv[i][0, 1], face_uv[i][1, 1], face_uv[i][0, 2],
                    face_uv[i][1, 2]))
            elif meshcolor == 2:
                fid.write(
                    '%d %d %d\n' % (face_colors[i, 0] * 255, face_colors[i, 1] * 255, face_colors[i, 2] * 255))

    fid.close()


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


File = Union[str, Path]


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


class RGBSource(Dataset):
    """Reads RGB video frames from disk in various formats (image folder, video file)."""

    def __init__(self, base_path: File, file_list: List[File],
                 transform: Optional[Callable[[np.ndarray], np.ndarray]] = None):
        """
        :param base_path: The path to a folder of video frames (images).
        :param file_list: The list of the images that should be loaded (just the filenames).
        :param transform: (optional) A function that takes in an image (HWC format) and returns an image of the same
            format that will be applied when the images are loaded later.
        """
        if file_list is None:
            file_list = []

        self.base_path = base_path
        self.file_list = file_list
        self.transform = transform

        self._validate()

    def _validate(self):
        """
        Make sure the folder and files exist.
        Throws an exception if the folder or any images cannot be opened.
        """
        if not os.path.isdir(self.base_path):
            raise RuntimeError(f"Could not open the folder {self.base_path}.")

        for filename in self.file_list:
            file_path = os.path.join(self.base_path, filename)

            if not os.path.isfile(file_path):
                raise RuntimeError(f"Could not open the file '{file_path}' in the folder {self.base_path}")

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        image_path = os.path.join(self.base_path, self.file_list[index])

        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transform:
            image = self.transform(image)

        return image

    @staticmethod
    def from_video(video_path: File, output_path: File,
                   transform: Callable[[np.ndarray], np.ndarray] = None) -> 'RGBSource':
        """
        Create a folder of images from a video file.

        :param video_path: The path to the video.
        :param output_path: The path to save the frames to (does not have to created first).
        :param transform: (optional) A function that takes in an image (HWC format) and returns an image of the same
            format that will be applied when the images are loaded later.
        :return: A RGBSource that points to the newly created image folder.
        """
        if not os.path.exists(video_path):
            raise RuntimeError(f"Could not open video file at {video_path}.")

        if not os.path.isfile(video_path):
            raise RuntimeError(f"Excepted a video file, got a folder for the path {video_path}.")

        if not os.path.isdir(output_path):
            os.makedirs(output_path, exist_ok=False)

            frames = RGBSource._get_video_frames(video_path)
            file_list = []

            for i, frame in enumerate(frames):
                filename = f"{i:06d}.jpg"
                file_list.append(filename)

                frame_output_path = os.path.join(output_path, filename)
                cv2.imwrite(frame_output_path, frame)
        else:
            file_list = sorted(os.listdir(output_path))

            video = cv2.VideoCapture(video_path)

            if not video.isOpened():
                raise RuntimeError(f"Could not open video file at {video_path}.")

            num_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

            if len(file_list) != num_frames:
                raise RuntimeError(f"Expected {num_frames:,d} frames in {output_path}, but found {len(file_list):,d}.")

        return RGBSource(output_path, file_list, transform)

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

        # TODO: Have each dataset class either infer or retrieve this information.
        fps = float(video.get(cv2.CAP_PROP_FPS))
        width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        num_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

        video_metadata = VideoMetadata(video_path, width=width, height=height, num_frames=num_frames, fps=fps)
        print(video_metadata)

        i = 0

        while video.grab():
            has_frame, frame = video.retrieve()
            if has_frame:
                yield frame

                i += 1
                print(f"\r[{i:03d}:{num_frames:03d}] Loading video...", end='')
            else:
                print(f"has_frame == False")

        print()
        video.release()


class DatasetBase:
    class InvalidDatasetFormatError(Exception):
        pass

    required_files = []
    required_folders = []
    mask_folder = 'mask'

    def __init__(self, base_path: Union[str, Path], should_create_masks=False, overwrite_ok=False,
                 use_estimated_depth=False, use_estimated_pose=False):
        """
        :param base_path: The path to the dataset.
        :param should_create_masks: Whether instance segmentation masks should be automatically created if
            they do not already exist.
        :param overwrite_ok: Whether it is okay to overwrite existing depth maps and/or instance segmentation masks.
        :param use_estimated_depth: Whether estimated depth maps should be used instead of any
            included ground truth depth maps.
        :param use_estimated_pose: Whether estimated camera parameters should be used instead of any
            included ground truth camera parameters.
        """
        self.base_path = base_path
        self.should_create_masks = should_create_masks
        self.overwrite_ok = overwrite_ok
        self.use_estimated_depth = use_estimated_depth
        self.use_estimated_pose = use_estimated_pose

        self.rgb_provider = None
        self.depth_provider = None
        self.camera_matrix_provider = None
        self.camera_trajectory_provider = None

        self._validate_dataset(base_path)

    @classmethod
    def is_valid_folder_structure(cls, path):
        try:
            cls._validate_dataset(path)
            return True
        except DatasetBase.InvalidDatasetFormatError:
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

        for filename in os.listdir(base_path):
            file_path = os.path.join(base_path, filename)

            if os.path.isfile(file_path):
                files_to_find.discard(filename)
            elif os.path.isdir(file_path):
                if len(os.listdir(file_path)) == 0 and filename in folders_to_find:
                    raise DatasetBase.InvalidDatasetFormatError(f"Empty folder {filename} in {base_path}.")

                folders_to_find.discard(filename)

        if len(files_to_find) > 0:
            raise DatasetBase.InvalidDatasetFormatError(
                f"Could not find the following required files {files_to_find} in {base_path}.")

        if len(folders_to_find) > 0:
            raise DatasetBase.InvalidDatasetFormatError(
                f"Could not find the following required folders {folders_to_find} in {base_path}.")


class TUMDataLoader(DatasetBase):
    """
    Loads image, depth and pose data from a TUM formatted dataset.
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

    def __init__(self, base_path, is_16_bit=True, frame_sampler=FrameSampler(),
                 should_create_masks=False, overwrite_ok=False, use_estimated_depth=False, use_estimated_pose=False):
        """
        :param base_path: The path to folder containing the dataset.
        :param is_16_bit: Whether the images are stored with 16-bit values or 32-bit values.
        :param frame_sampler: The frame sampler which chooses which frames to keep or discard.
        :param should_create_masks: Whether instance segmentation masks should be automatically created if
            they do not already exist.
        :param overwrite_ok: Whether it is okay to overwrite existing depth maps and/or instance segmentation masks.
        :param use_estimated_depth: Whether estimated depth maps should be used instead of any
            included ground truth depth maps.
        :param use_estimated_pose: Whether estimated camera parameters should be used instead of any
            included ground truth camera parameters.
        """
        super().__init__(base_path, should_create_masks=should_create_masks, overwrite_ok=overwrite_ok,
                         use_estimated_depth=use_estimated_depth, use_estimated_pose=use_estimated_pose)

        self.base_path = Path(base_path)
        self.pose_path = Path(os.path.join(base_path, str(Path(self.pose_path))))
        self.rgb_files_path = Path(os.path.join(base_path, str(Path(self.rgb_files_path))))
        self.depth_map_files_path = Path(os.path.join(base_path, str(Path(self.depth_map_files_path))))
        self.mask_path = None

        self.frame_sampler = frame_sampler
        self.is_16_bit = is_16_bit
        # The depth maps need to be divided by 5000 for the 16-bit PNG files
        # or 1.0 (i.e. no effect) for the 32-bit float images in the ROS bag files
        self.depth_scale_factor = 1.0 / 5000.0 if is_16_bit else 1.0

        self.synced_frame_data = None
        self.subset_indices = None
        self.rgb_frames = None
        self.depth_maps = None
        self.camera_trajectory = None
        self.masks = None

    @property
    def num_frames(self):
        return len(self.rgb_frames) if self.rgb_frames is not None else 0

    @property
    def camera_matrix(self):
        return TUMDataLoader.intrinsic_matrix.copy()

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
        # Convert paths to Path objects to ensure cross compatibility between operating systems.
        image_filenames_subset = map(Path, image_filenames_subset)

        depth_map_subset = depth_map_paths.flatten()
        depth_map_subset = map(Path, depth_map_subset)

        # Select the matching trajectory readings.
        trajectory_indices = get_match_indices(trajectory_timestamps, depth_map_timestamps)
        trajectory_subset = trajectory_data[trajectory_indices]

        def process_trajectory_datum(datum):
            tx, ty, tz, qx, qy, qz, qw = map(float, datum)
            r = Rotation.from_quat((qx, qy, qz, qw)).as_rotvec().reshape((-1, 1))
            t = np.array([tx, ty, tz]).reshape((-1, 1))
            pose = np.vstack((r, t))

            return pose

        trajectory_subset = np.array(list(map(process_trajectory_datum, trajectory_subset)))

        # # Rearrange pairs into the shape (N, 3) where N is the number of image and depth map pairs.
        # synced_frame_data = list(zip(image_filenames_subset, depth_map_subset, trajectory_subset))
        #
        # return synced_frame_data
        image_filenames_subset = list(
            map(lambda path: os.path.join(*map(str, (self.base_path, path))), image_filenames_subset))
        depth_map_subset = list(map(lambda path: os.path.join(*map(str, (self.base_path, path))), depth_map_subset))

        return image_indices, (image_filenames_subset, depth_map_subset, trajectory_subset)

    def load(self):
        """
        Load the data.
        :return: A 4-tuple containing the frames, depth maps, camera parameters and camera poses.
        """
        # TODO: Convert log to Timer.split
        log("Getting synced frame data...")
        self.subset_indices, self.synced_frame_data = self._get_synced_frame_data()

        frame_sampler = self.frame_sampler

        selected_frame_data = frame_sampler.choose(self.synced_frame_data)
        rgb_paths, depth_paths, poses = selected_frame_data
        log(f"Selected {len(rgb_paths)} frames.")

        log("Loading dataset...")

        pool = ThreadPool(psutil.cpu_count(logical=False))

        # TODO: Get rgb image path from the metadata
        rgb_folder = os.path.join(self.base_path, 'rgb')

        if self.use_estimated_depth:
            # TODO: Make the estimated depth folder configurable.
            estimated_depth_folder = 'estimated_depth'
            estimated_depth_path = os.path.join(self.base_path, estimated_depth_folder)

            if os.path.isdir(estimated_depth_path) and len(os.listdir(estimated_depth_path)) > 0:
                num_estimated_depth_maps = len(os.listdir(estimated_depth_path))
                num_frames = len(os.listdir(rgb_folder))

                if num_estimated_depth_maps == num_frames:
                    log(f"Found estimated depth maps in {estimated_depth_path}.")
                else:
                    raise RuntimeError(f"Found estimated depth maps in {estimated_depth_path} but found "
                                       f"{num_estimated_depth_maps} when {num_frames} was expected. "
                                       f"Potential fix: Delete the folder {estimated_depth_path} and run the program "
                                       f"again.")
            else:
                log("Estimating depth maps...")
                output_path = estimated_depth_path
                # TODO: Get the weights path from a config file pointing to the location in the Docker image.
                adabins_inference = InferenceHelper(weights_path='/root/.cache/pretrained')
                adabins_inference.predict_dir(rgb_folder, output_path)

            assert os.path.isdir(estimated_depth_path) and len(os.listdir(estimated_depth_path)) == len(
                os.listdir(estimated_depth_path))
            # Depth estimation models will use the rgb image names... so the paths will be identical to the rgb paths
            # except the folder will be 'estimated_depth' instead of 'rgb'.
            depth_paths = list(pool.map(lambda path: path.replace("rgb/", f"{estimated_depth_folder}/"), rgb_paths))
            log(f"Convert depth paths to estimated depth paths.")

        rgb_frames = pool.map(lambda path: cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB), rgb_paths)
        depth_maps = pool.map(lambda path: cv2.imread(path, cv2.IMREAD_ANYDEPTH), depth_paths)
        log(f"Loaded {len(rgb_frames)} frames.")

        rgb_frames = np.asarray(rgb_frames)
        depth_maps = np.asarray(depth_maps)
        log(f"Convert frame data to NumPy arrays.")

        if self.use_estimated_depth:
            # Assuming NYU formatted depth.
            # TODO: Make depth scaling factor configurable for estimated depth.
            depth_maps = depth_maps.astype(np.float32) / 1000.
        else:
            depth_maps = self.depth_scale_factor * depth_maps

        depth_maps = depth_maps.astype(np.float32)

        self.rgb_frames = rgb_frames
        self.depth_maps = depth_maps
        self.camera_trajectory = np.vstack(poses).reshape((-1, 6))

        mask_dir_hash = sha256(f"{self.base_path}{frame_sampler.start:06d}{frame_sampler.stop:06}"
                               f"{frame_sampler.stop_is_inclusive}".encode('utf-8')).hexdigest()
        mask_folder = os.path.join(self.base_path, mask_dir_hash)

        # TODO: Push this logic down to `create_masks(...)` for here and the similar code in `load_input_data(...)`.
        if os.path.isdir(mask_folder) and \
                len(os.listdir(mask_folder)) == len(rgb_frames):
            print(f"Found cached masks in {mask_folder}")
        elif self.should_create_masks:
            rgb_loader = DataLoader(NumpyDataset(rgb_frames),
                                    batch_size=8, shuffle=False)
            create_masks(rgb_loader, mask_folder, overwrite_ok=self.overwrite_ok)
        else:
            raise RuntimeError(f"Masks not found in path {mask_folder} or number of masks do not match the "
                               f"number of rgb frames in the selected set.")

        mask_paths = list(map(lambda filename: os.path.join(mask_folder, filename), sorted(os.listdir(mask_folder))))

        assert len(mask_paths) == len(rgb_frames)

        masks = pool.map(lambda path: cv2.imread(path, cv2.IMREAD_GRAYSCALE), mask_paths)
        masks = np.asarray(masks)

        log(f"Loaded {len(masks)} masks from {mask_folder}")

        self.masks = masks
        self.mask_path = mask_folder

        return self

    def get_info(self):
        image_resolution = "%dx%d" % (
            self.rgb_frames[0].shape[1], self.rgb_frames[0].shape[0]) if self.rgb_frames is not None else 'N/A'
        depth_map_resolution = "%dx%d" % (
            self.depth_maps[0].shape[1], self.depth_maps[0].shape[0]) if self.rgb_frames is not None else 'N/A'

        lines = [
            f"Dataset Info:",
            f"\tPath: {self.base_path}",
            f"\tTrajectory Data Path: {self.pose_path}",
            f"\tRGB Frame List Path: {self.rgb_files_path}",
            f"\tDepth Map List Path: {self.depth_map_files_path}",
            f"",
            f"\tTotal Num. Frames: {self.num_frames}",
            f"\tImage Resolution: {image_resolution}",
            f"\tDepth Map Resolution: {depth_map_resolution}",
            f"\tIs 16-bit: {self.is_16_bit}",
            f"\tDepth Scale: {self.depth_scale_factor:.4f}",
        ]

        return '\n'.join(lines)


class COLMAPProcessor:
    """
    Estimates camera trajectory and intrinsic parameters via COLMAP.
    """

    def __init__(self, storage_options: StorageOptions, colmap_options: COLMAPOptions, colmap_mask_folder='masks'):
        self.storage_options = storage_options
        self.colmap_options = colmap_options
        self.mask_folder = colmap_mask_folder

    @property
    def workspace_path(self):
        return self.storage_options.colmap_path

    @property
    def image_path(self):
        return self.storage_options.colour_path

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

        return

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
                   '--single_camera', 1 if options.is_single_camera else 0,  # COLMAP expects 1 for True, 0 for False.
                   '--dense', 1 if options.dense else 0,
                   '--quality', options.quality]

        if self.mask_path is not None:
            assert os.path.isdir(self.mask_path), f"Could not open mask folder: {self.mask_path}."
            command += ['--mask_path', self.mask_path]

        command = list(map(str, command))

        return ' '.join(command) if return_as_string else command

    def load_camera_params(self, raw_pose=False):
        num_models = len(os.listdir(self.result_path))
        assert num_models == 1, f"COLMAP reconstructed {num_models} when 1 was expected."
        sparse_recon_path = os.path.join(self.result_path, '0')

        print(f"Reading camera parameters from {sparse_recon_path}...")
        cameras, images, points3d = read_model(sparse_recon_path, ext=".bin")

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
                r, _ = cv2.Rodrigues(image.qvec2rotmat())
                t = image.tvec.reshape(-1, 1)

                extrinsic.append(np.vstack((r, t)))
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

                r, _ = cv2.Rodrigues(R)

                extrinsic.append(np.vstack((r, t)))

        extrinsic = np.asarray(extrinsic).squeeze()

        print(f"Read extrinsic parameters for {len(extrinsic)} frames.")

        return intrinsic, extrinsic


Size = Tuple[int, int]


class StrayScannerDataset(DatasetBase):
    """A dataset captured with 'Stray Scanner' on an iOS device with a LiDAR sensor."""

    # The files needed for a valid dataset.
    video_filename = 'rgb.mp4'
    camera_matrix_filename = 'camera_matrix.csv'
    camera_trajectory_filename = 'odometry.csv'
    required_files = [video_filename, camera_matrix_filename, camera_trajectory_filename]

    depth_folder = 'depth'
    confidence_map_folder = 'confidence'
    required_folders = [depth_folder, confidence_map_folder]

    depth_confidence_levels = [0, 1, 2]

    def __init__(self, base_path: Union[str, Path],
                 resize_to: Optional[Union[int, Size]] = None, depth_confidence_filter_level=0,
                 should_create_masks=False, overwrite_ok=False,
                 use_estimated_depth=False, use_estimated_pose=False):
        """
        :param base_path: The path to the dataset.
        :param resize_to: The resolution (height, width) to resize the images to.
        :param depth_confidence_filter_level: The minimum confidence value (0, 1, or 2) for the corresponding depth
                                              value to be kept. E.g. if set to 1, all pixels in the depth map where the
                                              corresponding pixel in the confidence map is less than 1 will be ignored.
        :param should_create_masks: Whether instance segmentation masks should be automatically created if
            they do not already exist.
        :param overwrite_ok: Whether it is okay to overwrite existing depth maps and/or instance segmentation masks.
        :param use_estimated_depth: Whether estimated depth maps should be used instead of any
            included ground truth depth maps.
        :param use_estimated_pose: Whether estimated camera parameters should be used instead of any
            included ground truth camera parameters.
        """
        super().__init__(base_path, should_create_masks=should_create_masks, overwrite_ok=overwrite_ok,
                         use_estimated_depth=use_estimated_depth, use_estimated_pose=use_estimated_pose)

        self.camera_matrix: Optional[np.ndarray] = None
        self.camera_trajectory: Optional[np.ndarray] = None
        self.rgb_frames: Optional[np.ndarray] = None
        self.depth_maps: Optional[np.ndarray] = None
        self.masks: Optional[np.ndarray] = None

        self.target_resolution = resize_to
        self.depth_confidence_filter_level = depth_confidence_filter_level

    def load(self):
        """
        Load the camera parameters and RGB-D data.

        :return: A reference to this object with the loaded data.
        """
        source_hw = self._get_frame_size()

        target_resolution = self.target_resolution

        if isinstance(target_resolution, int):
            # Cast results to int to avoid warning highlights in IDE.
            longest_side = int(np.argmax(source_hw))
            shortest_side = int(np.argmin(source_hw))

            new_size = [0, 0]
            new_size[longest_side] = target_resolution

            scale_factor = target_resolution / source_hw[longest_side]
            new_size[shortest_side] = source_hw[shortest_side] * scale_factor

            target_resolution = new_size
        elif isinstance(target_resolution, tuple):
            assert len(target_resolution) == 2, \
                f"The target resolution must be a 2-tuple, but got a {len(target_resolution)}-tuple."
            assert isinstance(target_resolution[0], int) and isinstance(target_resolution[1], int), \
                f"Expected target resolution to be a 2-tuple of integers, but got a tuple of" \
                f" ({type(target_resolution[0])}, {type(target_resolution[1])})."
        elif target_resolution is None:
            target_resolution = source_hw

        print(f"Resizing images (height, width) from {source_hw} to {target_resolution}.")

        target_orientation = 'portrait' if np.argmax(target_resolution) == 0 else 'landscape'
        source_orientation = 'portrait' if np.argmax(source_hw) == 0 else 'landscape'

        if target_orientation != source_orientation:
            warnings.warn(
                f"The input images appear to be in {source_orientation} ({source_hw[1]}x{source_hw[0]}), "
                f"but they are being resized to what appears to be "
                f"{target_orientation} ({target_resolution[1]}x{target_resolution[0]})")

        self.camera_matrix = self._load_camera_matrix(scale_x=target_resolution[1] / source_hw[1],
                                                      scale_y=target_resolution[0] / source_hw[0])
        self.camera_trajectory = self._load_camera_trajectory()

        self.rgb_frames = self._load_rgb(target_resolution=target_resolution)
        self.depth_maps = self._load_depth(target_resolution=target_resolution,
                                           filter_level=self.depth_confidence_filter_level)

        mask_path = os.path.join(self.base_path, self.mask_folder)
        num_frames = len(self.rgb_frames)

        if os.path.isdir(mask_path) and \
                len(os.listdir(mask_path)) == num_frames:
            print(f"Found cached masks in {mask_path}")
        elif self.should_create_masks:
            rgb_loader = DataLoader(NumpyDataset(self.rgb_frames),
                                    batch_size=8, shuffle=False)
            create_masks(rgb_loader, mask_path, overwrite_ok=self.overwrite_ok)
        else:
            raise RuntimeError(f"Masks not found in path {mask_path} or number of masks do not match the "
                               f"number of rgb frames in the selected set.")

        num_masks = len(os.listdir(mask_path))
        assert num_masks == num_frames, f"Expected to have the same number of RGB frames and masks, " \
                                        f"but found {num_frames} frames and {num_masks} mask."

        self.masks = self._load_mask(target_resolution=target_resolution)

        return self

    def _get_frame_size(self):
        """
        Get the resolution of the RGB video frames.

        :return: The frame resolution as a 2-tuple containing the height and width, respectively.
        """
        video_path = os.path.join(self.base_path, self.video_filename)
        video = cv2.VideoCapture(video_path)

        if not video.isOpened():
            raise RuntimeError(f"Could not open RGB video file: {video_path}")

        width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

        video.release()

        return height, width

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
            # x, y, z, qx, qy, qz, qw
            position = line[2:5]
            quaternion = line[5:]

            r = Rotation.from_quat(quaternion).as_rotvec()
            t = position

            pose = np.concatenate((r, t))
            trajectory.append(pose)

        trajectory = np.ascontiguousarray(trajectory)

        return trajectory

    def _load_rgb(self, target_resolution: Size):
        """
        Load the RGB frames.

        :param target_resolution: The resolution (height, width) to resize the images to.
        :return: A tensor containing all the RGB frames in NHWC format.
        """

        def rgb_transform(frame):
            frame = cv2.resize(frame, target_resolution)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # Input video is rotated 90 degrees anti-clockwise for some reason.
            # Need to adjust camera matrix and RGB-D data so that it is the right way up.
            frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)

            return frame

        rgb_dataset = RGBSource.from_video(
            os.path.join(self.base_path, self.video_filename),
            os.path.join(self.base_path, 'rgb'),
            transform=rgb_transform)

        pool = ThreadPool(processes=psutil.cpu_count(logical=False))

        start = time.time()
        frames = pool.map(lambda i: rgb_dataset[i], range(len(rgb_dataset)))
        elapsed = time.time() - start
        print(f"Took {elapsed:.2f}s to load {len(frames):,d} frames.")
        frames = np.ascontiguousarray(frames)

        return frames

    def _load_depth(self, target_resolution: Size, filter_level=0):
        """
        Load the depth maps.

        :param target_resolution: The resolution (height, width) to resize the images to.
        :param filter_level: The minimum confidence value (0, 1, or 2) for the corresponding depth value to be kept.
                             E.g. if set to 1, all pixels in the depth map where the corresponding pixel in the
                             confidence map is less than 1 will be ignored.
        :return: A tensor containing all the depth maps in NHW format.
        """
        assert filter_level in self.depth_confidence_levels, \
            f"Confidence filter must be one of the following: {self.depth_confidence_levels}."

        depth_path = os.path.join(self.base_path, self.depth_folder)
        confidence_path = os.path.join(self.base_path, self.confidence_map_folder)

        def load_depth_map(filename):
            # Code adapted from https://github.com/kekeblom/StrayVisualizer/blob/df5f39c750e8eec62b130dc9c8a91bdbcff1d952/stray_visualize.py#L58
            depth_map_path = os.path.join(depth_path, filename)
            depth_map_mm = np.load(depth_map_path)
            depth_map_m = depth_map_mm / 1000.0

            confidence_filename = f"{Path(filename).stem}.png"
            confidence_map_path = os.path.join(confidence_path, confidence_filename)
            confidence_map = cv2.imread(confidence_map_path, cv2.IMREAD_GRAYSCALE)

            depth_map_m[confidence_map < filter_level] = 0.0

            depth_map_m = cv2.resize(depth_map_m, target_resolution)
            # Input video is rotated 90 degrees anti-clockwise for some reason.
            # Need to adjust camera matrix and RGB-D data so that it is the right way up.
            depth_map_m = cv2.rotate(depth_map_m, cv2.ROTATE_90_CLOCKWISE)

            return depth_map_m

        pool = ThreadPool(psutil.cpu_count(logical=False))
        depth_maps = pool.map(load_depth_map, sorted(os.listdir(depth_path)))
        depth_maps = np.ascontiguousarray(depth_maps)

        return depth_maps

    def _load_mask(self, target_resolution: Size):
        """
        Load the instance segmentation masks.

        :param target_resolution: The resolution (height, width) to resize the masks to.
        :return: A tensor containing all the masks in NHW format.
        """
        # Input video is rotated 90 degrees anti-clockwise for some reason.
        # Need to adjust target_resolution so that it is the right way up.
        target_resolution = tuple(reversed(target_resolution))

        mask_path = os.path.join(self.base_path, self.mask_folder)

        def load_mask(filename):
            path = os.path.join(mask_path, filename)
            mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            mask = cv2.resize(mask, target_resolution)

            return mask

        pool = ThreadPool(psutil.cpu_count(logical=False))
        masks = pool.map(load_mask, sorted(os.listdir(mask_path)))
        masks = np.ascontiguousarray(masks)

        return masks
