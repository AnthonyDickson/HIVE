from hashlib import sha256

from pathlib import Path

import os
import struct
from multiprocessing.pool import ThreadPool

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

from Video2mesh.utils import Timer, log
from thirdparty.AdaBins.infer import InferenceHelper


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
    camera_params = np.loadtxt(os.path.join(storage_options.base_folder, "camera.txt"))
    camera_trajectory = np.loadtxt(os.path.join(storage_options.base_folder, "trajectory.txt"))

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

    if os.path.isdir(storage.mask_folder) and \
            len(os.listdir(storage.mask_folder)) == len(os.listdir(storage.colour_folder)):
        print(f"Found cached masks in {storage.mask_folder}")
    elif should_create_masks:
        rgb_loader = DataLoader(ImageFolderDataset(storage.colour_folder),
                                batch_size=batch_size, shuffle=False)
        create_masks(rgb_loader, storage.mask_folder, overwrite_ok=storage.overwrite_ok)
    else:
        raise RuntimeError(f"Masks not found in path {storage.mask_folder} or number of masks do not match the "
                           f"number of rgb frames in {storage.colour_folder}.")

    timer.split('locate/create masks')

    pool = ThreadPool(psutil.cpu_count(logical=False))

    rgb_frames = pool.map(cv2.imread, ImageFolderDataset(storage.colour_folder).image_paths)

    depth_maps = pool.map(lambda path: cv2.imread(path, cv2.IMREAD_ANYDEPTH),
                          ImageFolderDataset(storage.depth_folder).image_paths)

    masks = pool.map(lambda path: cv2.imread(path, cv2.IMREAD_GRAYSCALE),
                     ImageFolderDataset(storage.mask_folder).image_paths)

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


def create_masks(rgb_loader, mask_folder, overwrite_ok=False):
    """
    Create instance segmentation masks for the given RGB video sequence and save the masks to disk..

    :param rgb_loader: The PyTorch DataLoader that loads the RGB frames (no data augmentations applied).
    :param mask_folder: The path to save the masks to.
    :param overwrite_ok: Whether it is okay to write over any mask files in `mask_folder` if it already exists.
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
    max_num_masks = 0

    for image_batch in rgb_loader:
        outputs = predictor(image_batch.numpy())

        for output in outputs:
            matching_masks = output['instances'].get('pred_classes') == person_label
            people_masks = output['instances'].get('pred_masks')[matching_masks]
            combined_masks = np.zeros_like(image_batch[0].numpy(), dtype=np.uint8)
            combined_masks = combined_masks[:, :, 0]

            for j, mask in enumerate(people_masks.cpu().numpy()):
                combined_masks[mask] = j + 1

            i += 1
            max_num_masks = max(max_num_masks, combined_masks.max())
            Image.fromarray(combined_masks).convert('L').save(os.path.join(mask_folder, f"{i:03d}.png"))

        print(f"{i:03,d}/{len(rgb_loader.dataset):03,d}")


class ImageFolderDataset(Dataset):
    def __init__(self, base_dir, transform=None):
        assert os.path.isdir(base_dir), f"Could not find the folder: {base_dir}"

        self.base_dir = base_dir
        self.transform = transform

        filenames = list(sorted(os.listdir(base_dir)))
        assert len(filenames) > 0, f"No files found in the folder: {base_dir}"

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


class TUMDataLoader:
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

    def __init__(self, base_dir, is_16_bit=True,
                 pose_path="groundtruth.txt", rgb_files_path="rgb.txt",
                 depth_map_files_path="depth.txt"):
        """
        :param base_dir: The path to folder containing the dataset.
        :param is_16_bit: Whether the images are stored with 16-bit values or 32-bit values.
        :param pose_path: The name/path of the file that contains the camera pose information.
        :param rgb_files_path: The name/path of the file that contains the mapping of timestamps to image file paths.
        :param depth_map_files_path: The name/path of the file that contains the mapping of timestamps to depth map paths.
        """
        self.base_dir = Path(base_dir)
        self.pose_path = Path(os.path.join(base_dir, str(Path(pose_path))))
        self.rgb_files_path = Path(os.path.join(base_dir, str(Path(rgb_files_path))))
        self.depth_map_files_path = Path(os.path.join(base_dir, str(Path(depth_map_files_path))))
        self.mask_path = None

        self.is_16_bit = is_16_bit
        # The depth maps need to be divided by 5000 for the 16-bit PNG files
        # or 1.0 (i.e. no effect) for the 32-bit float images in the ROS bag files
        self.depth_scale_factor = 1.0 / 5000.0 if is_16_bit else 1.0

        self.synced_frame_data = None
        self.frames = None
        self.depth_maps = None
        self.poses = None
        self.masks = None

        self._validate_dataset()

    def _validate_dataset(self):
        """
        Check whether the dataset is valid and the expected files are present.
        :raises RuntimeError if there are any issues with the dataset.
        """
        if not self.base_dir.is_dir() or not self.base_dir.exists():
            raise RuntimeError(
                "The following path either does not exist, could not be read or is not a folder: %s." % self.base_dir)

        for path in (self.pose_path, self.rgb_files_path, self.depth_map_files_path):
            if not path.exists() or not path.is_file():
                raise RuntimeError("The following file either does not exist or could not be read: %s." % path)

    @property
    def num_frames(self):
        return len(self.frames) if self.frames is not None else 0

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
            map(lambda path: os.path.join(*map(str, (self.base_dir, path))), image_filenames_subset))
        depth_map_subset = list(map(lambda path: os.path.join(*map(str, (self.base_dir, path))), depth_map_subset))

        return image_filenames_subset, depth_map_subset, trajectory_subset

    def load(self, storage_options, frame_sampler=FrameSampler(), should_create_masks=True, use_estimated_depth=False):
        """
        Load the data.
        :param frame_sampler: The frame sampler which chooses which frames to keep or discard.
        :return: A 4-tuple containing the frames, depth maps, camera parameters and camera poses.
        """
        # TODO: Convert log to Timer.split
        log("Getting synced frame data...")
        self.synced_frame_data = self._get_synced_frame_data()

        selected_frame_data = frame_sampler.choose(self.synced_frame_data)
        rgb_paths, depth_paths, poses = selected_frame_data
        log(f"Selected {len(rgb_paths)} frames.")

        log("Loading dataset...")

        pool = ThreadPool(psutil.cpu_count(logical=False))

        # TODO: Get rgb image path from the metadata
        rgb_folder = os.path.join(self.base_dir, 'rgb')

        if use_estimated_depth:
            # TODO: Make the estimated depth folder configurable.
            estimated_depth_folder = 'estimated_depth'
            estimated_depth_path = os.path.join(self.base_dir, estimated_depth_folder)

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

            assert os.path.isdir(estimated_depth_path) and len(os.listdir(estimated_depth_path)) == len(os.listdir(estimated_depth_path))
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

        if use_estimated_depth:
            # Assuming NYU formatted depth.
            # TODO: Make depth scaling factor configurable for estimated depth.
            depth_maps = depth_maps.astype(np.float32) / 1000.
        else:
            depth_maps = self.depth_scale_factor * depth_maps

        depth_maps = depth_maps.astype(np.float32)

        self.frames = rgb_frames
        self.depth_maps = depth_maps
        self.poses = np.vstack(poses).reshape((-1, 6))

        mask_dir_hash = sha256(f"{self.base_dir}{frame_sampler.start:06d}{frame_sampler.stop:06}"
                               f"{frame_sampler.stop_is_inclusive}".encode('utf-8')).hexdigest()
        mask_folder = os.path.join(self.base_dir, mask_dir_hash)

        # TODO: Push this logic down to `create_masks(...)` for here and the similar code in `load_input_data(...)`.
        if os.path.isdir(mask_folder) and \
                len(os.listdir(mask_folder)) == len(rgb_frames):
            print(f"Found cached masks in {mask_folder}")
        elif should_create_masks:
            class NumpyDataset(Dataset):
                def __init__(self, rgb_frames):
                    self.frames = rgb_frames

                def __getitem__(self, index):
                    return self.frames[index]

                def __len__(self):
                    return len(self.frames)

            rgb_loader = DataLoader(NumpyDataset(rgb_frames),
                                    batch_size=8, shuffle=False)
            create_masks(rgb_loader, mask_folder, overwrite_ok=storage_options.overwrite_ok)
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
            self.frames[0].shape[1], self.frames[0].shape[0]) if self.frames is not None else 'N/A'
        depth_map_resolution = "%dx%d" % (
            self.depth_maps[0].shape[1], self.depth_maps[0].shape[0]) if self.frames is not None else 'N/A'

        lines = [
            f"Dataset Info:",
            f"\tPath: {self.base_dir}",
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
