import argparse
import enum
from typing import Optional

import cv2

from video2mesh.types import File


class ReprMixin:
    """Mixin that provides a basic string representation for objects."""

    def __repr__(self):
        def format_key_value_pair(key):
            value = self.__getattribute__(key)

            if isinstance(value, str):
                return f"{key}='{value}'"
            else:
                return f"{key}={value}"

        return f"{self.__class__.__name__}({', '.join(list(map(format_key_value_pair, self.__dict__)))})"

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

    def __init__(self, dataset_path: File, output_path: File, overwrite_ok=False, no_cache=False):
        """
        :param dataset_path: Path to the folder containing the RGB and depth image folders.
        :param output_path: Where to save the outputs.
        :param overwrite_ok: Whether it is okay to replace old results.
        :param no_cache: Whether cached datasets/results should be ignored.
        """
        self.dataset_path = dataset_path
        self.output_path = output_path
        self.overwrite_ok = overwrite_ok
        self.no_cache = no_cache

    @staticmethod
    def add_args(parser: argparse.ArgumentParser):
        group = parser.add_argument_group('Storage Options')

        group.add_argument('--dataset_path', type=str,
                           help='Path to the folder containing the RGB and depth image folders.',
                           required=True)
        group.add_argument('--output_path', type=str, required=True,
                           help='Where to save the outputs.')
        group.add_argument('--overwrite_ok', help='Whether it is okay to replace old results.',
                           action='store_true')
        group.add_argument('--no_cache', help='Whether cached datasets/results should be ignored.',
                           action='store_true')

    @staticmethod
    def from_args(args) -> 'StorageOptions':
        return StorageOptions(dataset_path=args.dataset_path, output_path=args.output_path,
                              overwrite_ok=args.overwrite_ok, no_cache=args.no_cache)


class COLMAPOptions(Options):
    quality_choices = ('low', 'medium', 'high', 'extreme')

    def __init__(self, is_single_camera=True, dense=False, quality='low', binary_path='/usr/local/bin/colmap',
                 vocab_path='/root/.cache/colmap/vocab.bin'):
        self.binary_path = binary_path
        self.vocab_path = vocab_path
        self.is_single_camera = is_single_camera
        self.dense = dense
        self.quality = quality

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
        group.add_argument('--binary_path', type=str, help='The path to the COLMAP binary.',
                           default='/usr/local/bin/colmap')
        group.add_argument('--vocab_path', type=str,
                           help='The path to the COLMAP vocabulary file. Defaults to the vocab file included in the '
                                'Docker image.', default='/root/.cache/colmap/vocab.bin')

    @staticmethod
    def from_args(args: argparse.Namespace) -> 'COLMAPOptions':
        return COLMAPOptions(is_single_camera=not args.multiple_cameras, dense=args.dense, quality=args.quality,
                             binary_path=args.binary_path, vocab_path=args.vocab_path)

    def __eq__(self, other) -> bool:
        return (
                self.binary_path == other.binary_path and
                self.vocab_path == other.vocab_path and
                self.is_single_camera == other.is_single_camera and
                self.dense == other.dense and
                self.quality == other.quality
        )

    def to_json(self) -> dict:
        """
        Convert the COLMAP configuration to a JSON friendly dictionary.
        :return: A dictionary containing the COLMAP configuration.
        """
        return dict(
            binary_path=self.binary_path,
            vocab_path=self.vocab_path,
            is_single_camera=self.is_single_camera,
            dense=self.dense,
            quality=self.quality
        )

    @classmethod
    def from_json(cls, json_dict: dict) -> 'COLMAPOptions':
        """
        Get a COLMAP configuration from a JSON dictionary.

        :param json_dict: A JSON formatted dictionary.
        :return: The COLMAP configuration.
        """
        return COLMAPOptions(
            binary_path=str(json_dict['binary_path']),
            vocab_path=str(json_dict['vocab_path']),
            is_single_camera=bool(json_dict['is_single_camera']),
            dense=bool(json_dict['dense']),
            quality=str(json_dict['quality']),
        )


class MeshDecimationOptions(Options):
    """Options for mesh decimation."""

    def __init__(self, num_vertices_background=2 ** 14, num_vertices_object=2 ** 10, max_error=0.001):
        """
        :param num_vertices_background: The target number of vertices for the background mesh.
            If set to -1, no mesh decimation is applied to the background.
        :param num_vertices_object: The target number of vertices for any object meshes.
            If set to -1, no mesh decimation is applied to the foreground meshes..
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

    def __init__(self, num_iterations=0, dilation_filter=cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))):
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
                           default=0)

    @staticmethod
    def from_args(args) -> 'MaskDilationOptions':
        return MaskDilationOptions(num_iterations=args.dilate_mask_iter)


class MeshFilteringOptions(Options):
    """Options for filtering mesh faces."""

    def __init__(self, max_pixel_distance=2, max_depth_distance=0.1, min_num_components=5):
        """
        :param max_pixel_distance: The maximum distance between vertices of a face in terms of their image space
        coordinates.
        :param max_depth_distance: The maximum difference in depth between vertices of a face.
        :param min_num_components: The minimum number of connected components in a mesh fragment.
        Fragments with fewer components will be culled.
        """
        self.max_pixel_distance = max_pixel_distance
        # Note: the default for max_depth_distance in the paper 'Soccer on Your Tabletop' is 0.02.
        #  However, the value of 0.1 seemed to produce better results with the TUM dataset.
        self.max_depth_distance = max_depth_distance
        self.min_num_components = min_num_components

    @staticmethod
    def add_args(parser: argparse.ArgumentParser):
        group = parser.add_argument_group('Mesh Filtering Options')

        group.add_argument('--max_depth_dist', type=float,
                           help='The maximum difference in depth between vertices of a '
                                'face. Used when filtering mesh faces.', default=0.1)
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


# noinspection PyArgumentList
class MeshReconstructionMethod(enum.Enum):
    # Uses a TSDF to reconstruct the 3D scene like KinectFusion, but poses must be known beforehand.
    TSDFFusion = enum.auto()
    # Similar to the TSDFFusion method, but run on small equal length subsequences instead of the entire video sequence.
    TSDFFusionChunks = enum.auto()
    # Uses a TSDF to reconstruct the 3D scene like KinectFusion, but uses 3D correspondence guided frame registration.
    BundleFusion = enum.auto()
    # Uses the first RGB-D frame to create background mesh.
    StaticRGBD = enum.auto()
    # Each frame will have a background mesh created from its RGB-D data
    RGBD = enum.auto()
    # Like the `RGBD` method except the background is only swapped out after the camera has moved enough.
    KeyframeRGBD = enum.auto()

    @classmethod
    def get_choices(cls):
        return {
            'tsdf_fusion': cls.TSDFFusion,
            'tsdf_fusion_chunks': cls.TSDFFusionChunks,
            'bundle_fusion': cls.BundleFusion,
            'static_rgbd': cls.StaticRGBD,
            'rgbd': cls.RGBD,
            'keyframe_rgbd': cls.KeyframeRGBD
        }

    @classmethod
    def from_string(cls, name):
        choices = cls.get_choices()

        if name.lower() in choices:
            return choices[name.lower()]
        else:
            raise RuntimeError(f"No method called {name}, valid choices are: {list(choices.keys())}")


class BackgroundMeshOptions(Options):
    supported_reconstruction_methods = [MeshReconstructionMethod.TSDFFusion, MeshReconstructionMethod.BundleFusion,
                                        MeshReconstructionMethod.StaticRGBD, MeshReconstructionMethod.RGBD]

    def __init__(self, reconstruction_method=MeshReconstructionMethod.TSDFFusion, depth_mask_dilation_iterations=32,
                 sdf_volume_size=5.0, sdf_voxel_size=0.02, sdf_max_voxels: Optional[int] = 80_000_000):
        """
        :param reconstruction_method: The method to use for reconstructing the background mesh(es).
        :param depth_mask_dilation_iterations: The number of times to dilate the dynamic object masks for masking the
            depth maps.
        :param sdf_volume_size: The size of the SDF volume in cubic meters. This option has no effect for the
            reconstruction method `tsdf_fusion` as it automatically infers the volume size from the input data.
        :param sdf_voxel_size: The size of a voxel in the SDF volume in cubic meters. Only applies to the reconstruction
            methods `tsdf_fusion` and `bundle_fusion`.
        :param sdf_max_voxels: (optional) The maximum number of voxels in the resulting voxel volume.
            This option only has an effect for the reconstruction method `tsdf_fusion`.
            If specified, the `sdf_voxel_size` option will be ignored.
        """
        assert reconstruction_method in BackgroundMeshOptions.supported_reconstruction_methods, \
            f"Reconstruction method must be one of the following: " \
            f"{[method.name for method in self.supported_reconstruction_methods]}, but got {reconstruction_method} " \
            f"instead."
        assert depth_mask_dilation_iterations >= 0 and isinstance(depth_mask_dilation_iterations, int), \
            f"The depth mask dilation iterations must be a positive integer."
        assert sdf_volume_size > 0.0, f"Volume size must be a positive number, instead got {sdf_volume_size}"
        assert sdf_voxel_size > 0.0, f"Voxel size must be a positive number, instead got {sdf_voxel_size}"
        assert sdf_max_voxels is None or (isinstance(sdf_max_voxels, int) and sdf_max_voxels > 0), \
            f"Number of voxels number must be a positive integer or None, instead got {sdf_max_voxels}"

        self.reconstruction_method = reconstruction_method
        self.depth_mask_dilation_iterations = depth_mask_dilation_iterations
        self.sdf_volume_size = sdf_volume_size
        self.sdf_voxel_size = sdf_voxel_size
        self.sdf_max_voxels = sdf_max_voxels

    @staticmethod
    def add_args(parser: argparse.ArgumentParser):
        group = parser.add_argument_group('Static Mesh Options')
        group.add_argument('--mesh_reconstruction_method', type=str,
                           help="The method to use for reconstructing the static mesh.",
                           choices=[method.name.lower() for method in
                                    BackgroundMeshOptions.supported_reconstruction_methods],
                           default='tsdf_fusion')
        group.add_argument('--depth_mask_dilation_iterations', type=int,
                           help="The number of times to dilate the dynamic object masks for masking the depth maps.",
                           default=32)
        group.add_argument('--sdf_volume_size', type=float,
                           help="The size of the SDF volume in cubic meters. This option has no effect for the "
                                "reconstruction method `TSDF_FUSION` as it automatically infers the volume "
                                "size from the input data.", default=5.0)
        group.add_argument('--sdf_voxel_size', type=float,
                           help="The size of a voxel in the SDF volume in cubic meters.", default=0.02)
        group.add_argument('--sdf_max_voxels', type=int, default=80_000_000,
                           help="The maximum number of voxels allowed in the resulting voxel volume."
                                "This option only has an effect for the reconstruction method `TSDF_FUSION`. "
                                "If specified, the `--sdf_voxel_size` option will be ignored.")

    @staticmethod
    def from_args(args: argparse.Namespace) -> 'BackgroundMeshOptions':
        return BackgroundMeshOptions(
            reconstruction_method=MeshReconstructionMethod.from_string(args.mesh_reconstruction_method),
            depth_mask_dilation_iterations=args.depth_mask_dilation_iterations,
            sdf_volume_size=args.sdf_volume_size,
            sdf_voxel_size=args.sdf_voxel_size,
            sdf_max_voxels=args.sdf_max_voxels
        )


class ForegroundTrajectorySmoothingOptions(Options):
    def __init__(self, learning_rate=1e-5, num_epochs=0):
        """
        :param learning_rate: The learning rate/step size to take each epoch when smoothing the trajectory.
        :param num_epochs: The number of iterations to loop the smoothing algorithm. Set to zero to disable
            foreground trajectory smoothing.
        """
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs

    @staticmethod
    def add_args(parser: argparse.ArgumentParser):
        group = parser.add_argument_group('Foreground Trajectory Smoothing')
        group.add_argument('--fts_learning_rate', type=float, default=1e-5,
                           help='The learning rate/step size to take each epoch when smoothing the trajectory. ')
        group.add_argument('--fts_num_epochs', type=int, default=0,
                           help='The number of iterations to loop the smoothing algorithm. Set to zero to disable '
                                'foreground trajectory smoothing.')

    @staticmethod
    def from_args(args: argparse.Namespace) -> 'ForegroundTrajectorySmoothingOptions':
        return ForegroundTrajectorySmoothingOptions(
            learning_rate=args.fts_learning_rate,
            num_epochs=args.fts_num_epochs
        )

class WebXROptions(Options):
    """Configuration for the WebXR renderer, and the metadata."""
    def __init__(self, webxr_path='thirdparty/webxr3dvideo/docs', webxr_url='localhost:8080',
                 webxr_add_ground_plane=False, webxr_add_sky_box=False):
        """
        :param webxr_path: Where to export the 3D video files to.
        :param webxr_url: The URL to the WebXR 3D video player.
        :param webxr_add_ground_plane: Whether to render a white ground plane to the scene in the renderer. Useful for debugging.
        :param webxr_add_sky_box: Whether to render a sky as a 360 degree background (cube map).
        """
        self.webxr_path = webxr_path
        self.webxr_url = webxr_url
        self.webxr_add_ground_plane = webxr_add_ground_plane
        self.webxr_add_sky_box = webxr_add_sky_box

    @staticmethod
    def add_args(parser: argparse.ArgumentParser):
        group = parser.add_argument_group('WebXR')

        group.add_argument('--webxr_path', type=str, help='Where to export the 3D video files to.',
                           default='thirdparty/webxr3dvideo/docs')
        group.add_argument('--webxr_url', type=str, help='The URL to the WebXR 3D video player.',
                           default='http://localhost:8080')
        group.add_argument('--webxr_add_ground_plane', action='store_true',
                           help='Whether to render a white ground plane to the scene in the renderer.')
        group.add_argument('--webxr_add_sky_box', action='store_true',
                           help='Whether to render a sky cube map in the background.')

    @staticmethod
    def from_args(args: argparse.Namespace) -> 'WebXROptions':
        return WebXROptions(
            webxr_path=args.webxr_path,
            webxr_url=args.webxr_url,
            webxr_add_ground_plane=args.webxr_add_ground_plane,
            webxr_add_sky_box=args.webxr_add_sky_box
        )

class PipelineOptions(Options):

    def __init__(self,
                 num_frames=-1, frame_step=15,
                 estimate_pose=False, estimate_depth=False,
                 background_only=False, align_scene=False,
                 log_file='logs.log', use_lama=False):
        """
        :param num_frames: The maximum of frames to process. Set to -1 (default) to process all frames.
        :param frame_step: The frequency to sample frames at for COLMAP and pose optimisation.
            If set to 1, samples all frames (i.e. no effect). Otherwise if set to n > 1, samples every n frames.
        :param estimate_pose: Whether to estimate camera parameters with COLMAP or use provided ground truth data.
        :param estimate_depth: Whether to estimate depth maps or use provided ground truth depth maps.
        :param background_only: Whether to only reconstruct the static background.
        :param align_scene: Whether to align the scene with the ground plane. Enable this if the recording device was held at an angle (facing upwards or downwards, not level) and the scene is not level in the renderer.
        :param log_file: The path to save the logs to.
        :param use_lama: Include lama inpainting in the pipeline process.
        """
        self.num_frames = num_frames
        self.frame_step = frame_step
        self.estimate_pose = estimate_pose
        self.estimate_depth = estimate_depth
        self.background_only = background_only
        self.align_scene = align_scene
        self.log_file = log_file
        self.use_lama = use_lama

    @staticmethod
    def add_args(parser: argparse.ArgumentParser):
        group = parser.add_argument_group('Pipeline')

        group.add_argument('--num_frames', type=int, default=-1,
                           help='The maximum of frames to process. Set to -1 (default) to process all frames.')
        group.add_argument('--frame_step', type=int, default=15,
                           help='The frequency to sample frames at for COLMAP and pose optimisation. '
                                'If set to 1, samples all frames (i.e. no effect). '
                                'Otherwise if set to n > 1, samples every n frames.')
        group.add_argument('--estimate_pose', action='store_true',
                           help='Whether to estimate camera parameters with COLMAP or use provided ground truth data.')
        group.add_argument('--estimate_depth', action='store_true',
                           help='Whether to estimate depth maps or use provided ground truth depth maps.')
        group.add_argument('--background_only', action='store_true',
                           help='Whether to only reconstruct the static background.')
        group.add_argument('--align_scene', action='store_true',
                           help='Whether to align the scene with the ground plane. Enable this if the recording device was held at an angle (facing upwards or downwards, not level) and the scene is not level in the renderer.')
        group.add_argument('--log_file', type=str, help='The path to save the logs to.',
                           default='logs.log')
        group.add_argument('--use_lama', help='Whether to use lama inpainting in the pipeline process.',
                           action='store_true')

    @staticmethod
    def from_args(args: argparse.Namespace) -> 'PipelineOptions':
        return PipelineOptions(
            num_frames=args.num_frames,
            frame_step=args.frame_step,
            estimate_pose=args.estimate_pose,
            estimate_depth=args.estimate_depth,
            background_only=args.background_only,
            align_scene=args.align_scene,
            log_file=args.log_file,
            use_lama=args.use_lama
        )
