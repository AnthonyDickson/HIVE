import warnings

import argparse
from typing import Optional

import cv2
import enum
import numpy as np


class ReprMixin:
    """Mixin that provides a basic string representation for objects."""

    def __repr__(self):
        return f"{self.__class__.__name__}({', '.join(list(map(lambda k: f'{k}={self.__getattribute__(k)}', self.__dict__)))})"

    def __str__(self):
        return repr(self)


class Options(ReprMixin):
    """
    Interface for objects that store options that can be initialised either programmatically or
    via command-line arguments.
    """

    @staticmethod
    def add_args(parser: argparse.ArgumentParser):
        """
        Add arguments to a parser (modifies object in-place).
        Implementing members should add the new arguments to a group.

        :param parser: The parser object to add the arguments to.
        """
        raise NotImplementedError

    @staticmethod
    def from_args(args: argparse.Namespace) -> 'Options':
        """
        Create an Options object from parsed command line arguments.

        :param args: The namespace object from calling `parser.parse_args()`.
        """
        raise NotImplementedError


class StorageOptions(Options):
    """Options regarding storage of inputs and outputs."""

    def __init__(self, base_path, overwrite_ok=False):
        """
        :param base_path: Path to the folder containing the RGB and depth image folders.'
        :param overwrite_ok: Whether it is okay to replace old results.
        """
        self.base_path = base_path
        self.overwrite_ok = overwrite_ok

    @staticmethod
    def add_args(parser: argparse.ArgumentParser):
        group = parser.add_argument_group('Storage Options')

        group.add_argument('--base_path', type=str,
                           help='Path to the folder containing the RGB and depth image folders.',
                           required=True)
        group.add_argument('--overwrite_ok', help='Whether it is okay to replace old results.',
                           action='store_true')

    @staticmethod
    def from_args(args) -> 'StorageOptions':
        return StorageOptions(base_path=args.base_path, overwrite_ok=args.overwrite_ok)


class DepthFormat(enum.Enum):
    DEPTH_TO_POINT = enum.auto()
    DEPTH_TO_PLANE = enum.auto()


class DepthEstimationModel(enum.Enum):
    ADABINS = enum.auto()
    LERES = enum.auto()
    CVDE = enum.auto()

    @classmethod
    def get_choices(cls):
        return {
            'adabins': cls.ADABINS,
            'leres': cls.LERES,
            'cvde': cls.CVDE
        }

    @classmethod
    def from_string(cls, name):
        choices = cls.get_choices()

        if name.lower() in choices:
            return choices[name.lower()]
        else:
            raise RuntimeError(f"No model called {name}, valid choices are: {list(choices.keys())}")


class DepthOptions(Options):
    """Options for depth maps."""

    supported_depth_estimation_models = [DepthEstimationModel.ADABINS, DepthEstimationModel.LERES,
                                         DepthEstimationModel.CVDE]

    def __init__(self, max_depth=10.0, dtype=np.uint16, depth_format=DepthFormat.DEPTH_TO_PLANE,
                 depth_estimation_model=DepthEstimationModel.ADABINS, sampling_framerate=-1):
        """
        :param max_depth: The maximum depth value in the depth maps.
        :param dtype: The type of the depth values.
        :param depth_format: How depth values are measured in the depth maps.
        :param depth_estimation_model: The model to use to estimate depth maps.
        :param sampling_framerate: The number of frames to sample every second for the CVDE depth estimation method.
        """
        assert dtype is np.uint8 or dtype is np.uint16, 'Only 8-bit and 16-bit depth maps are supported.'

        assert depth_estimation_model in DepthOptions.supported_depth_estimation_models, \
            f"Depth estimation model must be one of the following: " \
            f"{[model.name for model in self.supported_depth_estimation_models]}, but got {depth_estimation_model.name} " \
            f"instead."

        if not isinstance(max_depth, (int, float)) or not np.isfinite(max_depth) or max_depth <= 0.0:
            raise ValueError(f"Max depth must a finite, positive number, but got {max_depth}.")

        if max_depth != 10.0:
            warnings.warn("The --max_depth option has no effect in this version.")
            # TODO: Figure out a clean way to apply max depth and to ensure that results that used a different
            #  max depth are overwritten.

        if not isinstance(sampling_framerate, int) or (sampling_framerate < 1 and sampling_framerate != -1):
            raise ValueError(f"Sampling framerate must be a positive integer or -1, but got {sampling_framerate}.")

        self.max_depth = max_depth
        self.depth_dtype = dtype
        self.depth_format = depth_format
        self.depth_estimation_model = depth_estimation_model
        self.sampling_framerate = sampling_framerate

    @staticmethod
    def add_args(parser: argparse.ArgumentParser):
        group = parser.add_argument_group('Depth Options')
        group.add_argument('--max_depth', type=float, help='The maximum depth value in the provided depth maps.',
                           default=10.0)
        group.add_argument('--depth_format', type=str, help='How depth values are measure in the depth maps.',
                           choices=['depth_to_point', 'depth_to_plane'], default='depth_to_plane')
        group.add_argument('--depth_dtype', type=str, help='The type of the depth values.', default='uint16',
                           choices=['uint8', 'uint16'])
        group.add_argument('--depth_estimation_model', type=str,
                           help="The model to use for estimating depth maps.",
                           choices=[model.name.lower() for model in DepthOptions.supported_depth_estimation_models],
                           default=DepthEstimationModel.ADABINS.name.lower())
        group.add_argument('--sampling_framerate', type=int,
                           help='The number of frames to sample every second for the CVDE depth estimation method. '
                                'Defaults to every frame (this may be very slow depending on the number of frames!)',
                           default=-1)

    @staticmethod
    def from_args(args) -> 'DepthOptions':
        max_depth = args.max_depth
        dtype = args.depth_dtype
        depth_format = args.depth_format

        if dtype == 'uint8':
            dtype = np.uint8
        elif dtype == 'uint16':
            dtype = np.uint16
        else:
            raise RuntimeError(f"Unsupported data type {dtype}, expected 'uint8' or 'uint16")

        if depth_format == 'depth_to_point':
            depth_format = DepthFormat.DEPTH_TO_POINT
        elif depth_format == 'depth_to_plane':
            depth_format = DepthFormat.DEPTH_TO_PLANE
        else:
            raise RuntimeError(f"Unsupported depth format {depth_format}, "
                               f"expected 'depth_to_point' or 'depth_to_plane'")

        return DepthOptions(max_depth, dtype, depth_format,
                            DepthEstimationModel.from_string(args.depth_estimation_model), args.sampling_framerate)


class COLMAPOptions(Options):
    quality_choices = ('low', 'medium', 'high', 'extreme')

    def __init__(self, is_single_camera=True, dense=False, quality='high', use_raw_pose=False,
                 binary_path='/usr/local/bin/colmap', vocab_path='/root/.cache/colmap/vocab.bin'):
        self.binary_path = binary_path
        self.vocab_path = vocab_path
        self.is_single_camera = is_single_camera
        self.dense = dense
        self.quality = quality
        self.use_raw_pose = use_raw_pose

        assert quality in COLMAPOptions.quality_choices, f"Quality must be one of: {COLMAPOptions.quality_choices}, got {quality}."

    @staticmethod
    def add_args(parser: argparse.ArgumentParser):
        group = parser.add_argument_group('COLMAP Options')

        group.add_argument('--multiple_cameras', action='store_true',
                           help='Whether the video dataset was captured with multiple camera devices or '
                                'a single camera device with different settings per-frame (e.g. focal length).')
        group.add_argument('--dense', action='store_true', help='Whether to run dense reconstruction.')
        group.add_argument('--quality', type=str, help='The quality of the COLMAP reconstruction.',
                           default='low', choices=COLMAPOptions.quality_choices)
        group.add_argument('--use_raw_pose', action='store_true',
                           help='Whether to use the pose data straight from COLMAP, or to convert the pose data into a '
                                'more "normal" coordinate system.')
        group.add_argument('--binary_path', type=str, help='The path to the COLMAP binary.',
                           default='/usr/local/bin/colmap')
        group.add_argument('--vocab_path', type=str,
                           help='The path to the COLMAP vocabulary file. Defaults to the vocab file included in the '
                                'Docker image.', default='/root/.cache/colmap/vocab.bin')

    @staticmethod
    def from_args(args: argparse.Namespace) -> 'COLMAPOptions':
        return COLMAPOptions(
            binary_path=args.binary_path,
            vocab_path=args.vocab_path,
            is_single_camera=not args.multiple_cameras,
            dense=args.dense,
            quality=args.quality,
            use_raw_pose=args.use_raw_pose
        )


class MeshDecimationOptions(Options):
    """Options for mesh decimation."""

    def __init__(self, num_vertices_background=2 ** 14, num_vertices_object=2 ** 10, max_error=0.001):
        """
        :param num_vertices_background: The target number of vertices for the background mesh.
        :param num_vertices_object: The target number of vertices for any object meshes.
        :param max_error: Not sure what this parameter does exactly...
        """
        self.num_vertices_background = num_vertices_background
        self.num_vertices_object = num_vertices_object
        self.max_error = max_error

    @staticmethod
    def add_args(parser: argparse.ArgumentParser):
        group = parser.add_argument_group('Mesh Decimation Options')

        group.add_argument('--num_vertices_background', type=int,
                           help="The target number of vertices for the background mesh.", default=2 ** 14)
        group.add_argument('--num_vertices_object', type=int,
                           help="The target number of vertices for any object meshes.", default=2 ** 10)
        group.add_argument('--decimation_max_error', type=float, help="Not sure what this parameter does exactly...",
                           default=0.001)

    @staticmethod
    def from_args(args) -> 'MeshDecimationOptions':
        return MeshDecimationOptions(
            num_vertices_background=args.num_vertices_background,
            num_vertices_object=args.num_vertices_object,
            max_error=args.decimation_max_error,
        )


class MaskDilationOptions(Options):
    """Options for the function `dilate_mask`."""

    def __init__(self, num_iterations=3, dilation_filter=cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))):
        """
        :param num_iterations: The number of times to apply the dilation filter.
        :param dilation_filter: The filter to dilate with (e.g. cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)).
        """
        self.num_iterations = num_iterations
        self.filter = dilation_filter

    @staticmethod
    def add_args(parser: argparse.ArgumentParser):
        group = parser.add_argument_group('Mask Dilation Options')

        group.add_argument('--dilate_mask_iter', type=int,
                           help='The number of times to run a dilation filter over the '
                                'object masks. A higher number results in larger masks and '
                                'zero results in the original mask.',
                           default=3)

    @staticmethod
    def from_args(args) -> 'MaskDilationOptions':
        return MaskDilationOptions(num_iterations=args.dilate_mask_iter)


class MeshFilteringOptions(Options):
    """Options for filtering mesh faces."""

    def __init__(self, max_pixel_distance=2, max_depth_distance=0.02, min_num_components=5):
        """
        :param max_pixel_distance: The maximum distance between vertices of a face in terms of their image space
        coordinates.
        :param max_depth_distance: The maximum difference in depth between vertices of a face.
        :param min_num_components: The minimum number of connected components in a mesh fragment.
        Fragments with fewer components will be culled.
        """
        self.max_pixel_distance = max_pixel_distance
        self.max_depth_distance = max_depth_distance
        self.min_num_components = min_num_components

    @staticmethod
    def add_args(parser: argparse.ArgumentParser):
        group = parser.add_argument_group('Mesh Filtering Options')

        group.add_argument('--max_depth_dist', type=float,
                           help='The maximum difference in depth between vertices of a '
                                'face. Used when filtering mesh faces.', default=0.02)
        group.add_argument('--max_pixel_dist', type=float,
                           help='The maximum distance between vertices of a face in terms of their image space '
                                'coordinates.', default=2)
        group.add_argument('--min_num_components', type=float,
                           help='The minimum number of connected components in a mesh fragment. '
                                'Fragments with fewer components will be culled.', default=5)

    @staticmethod
    def from_args(args) -> 'MeshFilteringOptions':
        return MeshFilteringOptions(max_pixel_distance=args.max_pixel_dist,
                                    max_depth_distance=args.max_depth_dist,
                                    min_num_components=args.min_num_components)


class MeshReconstructionMethod(enum.Enum):
    TSDF_FUSION = enum.auto()
    BUNDLE_FUSION = enum.auto()

    @classmethod
    def get_choices(cls):
        return {
            'tsdf_fusion': cls.TSDF_FUSION,
            'bundle_fusion': cls.BUNDLE_FUSION
        }

    @classmethod
    def from_string(cls, name):
        choices = cls.get_choices()

        if name.lower() in choices:
            return choices[name.lower()]
        else:
            raise RuntimeError(f"No method called {name}, valid choices are: {list(choices.keys())}")


class StaticMeshOptions(Options):
    supported_reconstruction_methods = [MeshReconstructionMethod.TSDF_FUSION,
                                        MeshReconstructionMethod.BUNDLE_FUSION]

    def __init__(self, reconstruction_method=MeshReconstructionMethod.TSDF_FUSION, depth_mask_dilation_iterations=32,
                 sdf_volume_size=3.0, sdf_voxel_size=0.02, sdf_num_voxels: Optional[int] = None):
        """
        :param reconstruction_method: The method to use for reconstructing the static mesh.
        :param depth_mask_dilation_iterations: The number of times to dilate the dynamic object masks for masking the
            depth maps.
        :param sdf_volume_size: The size of the SDF volume in cubic meters. This option has no effect for the
            reconstruction method `TSDF_FUSION` as it automatically infers the volume size from the input data.
        :param sdf_voxel_size: The size of a voxel in the SDF volume in cubic meters.
        :param sdf_num_voxels: (optional) The desired number of voxels in the resulting voxel volume.
            This option only has an effect for the reconstruction method `TSDF_FUSION`.
            If specified, the `sdf_voxel_size` option will be ignored.
        """
        assert reconstruction_method in StaticMeshOptions.supported_reconstruction_methods, \
            f"Reconstruction method must be one of the following: " \
            f"{[method.name for method in self.supported_reconstruction_methods]}, but got {reconstruction_method} " \
            f"instead."
        assert depth_mask_dilation_iterations >= 0 and isinstance(depth_mask_dilation_iterations, int), \
            f"The depth mask dilation iterations must be a positive integer."
        assert sdf_volume_size > 0.0, f"Volume size must be a positive number, instead got {sdf_volume_size}"
        assert sdf_voxel_size > 0.0, f"Voxel size must be a positive number, instead got {sdf_voxel_size}"
        assert sdf_num_voxels is None or (isinstance(sdf_num_voxels, int) and sdf_num_voxels > 0), \
            f"Number of voxels number must be a positive integer or None, instead got {sdf_num_voxels}"

        self.reconstruction_method = reconstruction_method
        self.depth_mask_dilation_iterations = depth_mask_dilation_iterations
        self.sdf_volume_size = sdf_volume_size
        self.sdf_voxel_size = sdf_voxel_size
        self.sdf_num_voxels = sdf_num_voxels

    @staticmethod
    def add_args(parser: argparse.ArgumentParser):
        group = parser.add_argument_group('Static Mesh Options')
        group.add_argument('--mesh_reconstruction_method', type=str,
                           help="The method to use for reconstructing the static mesh.",
                           choices=[method.name.lower() for method in
                                    StaticMeshOptions.supported_reconstruction_methods],
                           default='tsdf_fusion')
        group.add_argument('--depth_mask_dilation_iterations', type=int,
                           help="The number of times to dilate the dynamic object masks for masking the depth maps.",
                           default=32)
        group.add_argument('--sdf_volume_size', type=float,
                           help="The size of the SDF volume in cubic meters. This option has no effect for the "
                                "reconstruction method `TSDF_FUSION` as it automatically infers the volume "
                                "size from the input data.", default=3.0)
        group.add_argument('--sdf_voxel_size', type=float,
                           help="The size of a voxel in the SDF volume in cubic meters.", default=0.02)
        group.add_argument('--sdf_num_voxels', type=int,
                           help="The desired number of voxels in the resulting voxel volume. "
                                "This option only has an effect for the reconstruction method `TSDF_FUSION`. "
                                "If specified, the `--sdf_voxel_size` option will be ignored.")

    @staticmethod
    def from_args(args: argparse.Namespace) -> 'StaticMeshOptions':
        return StaticMeshOptions(
            reconstruction_method=MeshReconstructionMethod.from_string(args.mesh_reconstruction_method),
            depth_mask_dilation_iterations=args.depth_mask_dilation_iterations,
            sdf_volume_size=args.sdf_volume_size,
            sdf_voxel_size=args.sdf_voxel_size,
            sdf_num_voxels=args.sdf_num_voxels
        )


class Video2MeshOptions(Options):

    def __init__(self, create_masks=False,
                 include_background=False, static_background=False,
                 num_frames=-1,
                 estimate_depth=False, estimate_camera_params=False,
                 refine_colmap_poses=False,
                 webxr_path='thirdparty/webxr3dvideo/docs', webxr_url='localhost:8080'):
        """
        :param create_masks: Whether to create masks for dynamic objects
        :param include_background: Include the background in the reconstructed mesh.
        :param static_background: Whether to use the first frame to generate a static background.
        :param num_frames: The maximum of frames to process. Set to -1 (default) to process all frames.
        :param estimate_depth: Flag to indicate that depth maps estimated by a neural network model should be used
                                instead of the ground truth depth maps.
        :param estimate_camera_params: Flag to indicate that camera intrinsic and extrinsic parameters estimated with
                                       COLMAP should be used instead of the ground truth parameters (if they exist).
        :param refine_colmap_poses: Whether to refine estimated pose data from COLMAP with pose data from BundleFusion.
            Note that this argument is only valid if the flag '--estimate_camera_params' is used and BundleFusion is the
            specified mesh reconstruction method.
        :param webxr_path: Where to export the 3D video files to.
        :param webxr_url: The URL to the WebXR 3D video player.
        """
        self.create_masks = create_masks
        self.include_background = include_background
        self.static_background = static_background
        self.num_frames = num_frames
        self.estimate_depth = estimate_depth
        self.estimate_camera_params = estimate_camera_params
        self.refine_colmap_poses = refine_colmap_poses
        self.webxr_path = webxr_path
        self.webxr_url = webxr_url

        if self.include_background:
            warnings.warn("The command line option `--include_background` is deprecated and will be removed in "
                          "future versions. KinectFusion will reconstruct the 3D background instead.")

        if self.static_background:
            warnings.warn("The command line option `--static_background` is deprecated and will be removed in "
                          "future versions. KinectFusion will reconstruct the 3D background instead.")

    @staticmethod
    def add_args(parser: argparse.ArgumentParser):
        group = parser.add_argument_group('video2mesh')

        group.add_argument('--create_masks', help='Whether to create masks for dynamic objects',
                           action='store_true')
        group.add_argument('--include_background', help='Include the background in the reconstructed mesh.',
                           action='store_true')
        group.add_argument('--static_background',
                           help='Whether to use the first frame to generate a static background.',
                           action='store_true')
        group.add_argument('--num_frames', type=int, help='The maximum of frames to process. '
                                                          'Set to -1 (default) to process all frames.', default=-1)
        group.add_argument('--estimate_depth', action='store_true',
                           help='Flag to indicate that depth maps estimated by a neural network model should be used '
                                'instead of the ground truth depth maps.')
        group.add_argument('--estimate_camera_params', action='store_true',
                           help='Flag to indicate that camera intrinsic and extrinsic parameters estimated with COLMAP '
                                'should be used instead of the ground truth parameters (if they exist).')
        group.add_argument('--refine_colmap_poses',
                           help="Whether to refine estimated pose data from COLMAP with pose data from BundleFusion. "
                                "Note that this argument is only valid if the flag '--estimate_camera_params' is used "
                                "and BundleFusion is the specified mesh reconstruction method.",
                           action='store_true')
        group.add_argument('--webxr_path', type=str, help='Where to export the 3D video files to.',
                           default='thirdparty/webxr3dvideo/docs')
        group.add_argument('--webxr_url', type=str, help='The URL to the WebXR 3D video player.',
                           default='http://localhost:8080')

    @staticmethod
    def from_args(args: argparse.Namespace) -> 'Video2MeshOptions':
        return Video2MeshOptions(
            create_masks=args.create_masks,
            include_background=args.include_background,
            static_background=args.static_background,
            num_frames=args.num_frames,
            estimate_depth=args.estimate_depth,
            estimate_camera_params=args.estimate_camera_params,
            refine_colmap_poses=args.refine_colmap_poses,
            webxr_path=args.webxr_path,
            webxr_url=args.webxr_url
        )
