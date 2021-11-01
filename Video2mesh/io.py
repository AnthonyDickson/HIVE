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
from torch.utils.data import DataLoader, Dataset

from Video2mesh.utils import Timer


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


def load_input_data(storage_options, batch_size=-1, should_create_masks=False, timer=Timer()):
    """
    Load RGB-D frames and instance segmentation masks from disk.

    :param storage_options: The options file that contains the path to the dataset.
    :param batch_size: How many files to load at once when creating masks. Has little impact on performance.
    :param should_create_masks: Whether instance segmentation masks should be created if they do not already exist.
    :param timer: Timer object for simple benchmarking of code segments.

    :return: A 3-tuple of the RGB frames (N, H, W, C), depth maps (N, H, W) and masks (N, H, W).
    """
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
    depth_maps = pool.map(lambda path: cv2.imread(path, cv2.IMREAD_GRAYSCALE),
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
