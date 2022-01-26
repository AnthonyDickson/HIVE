import argparse
import cv2
import enum
import numpy as np


class ReprMixin:
    """Mixin that provides a basic string representation for objects."""

    def __repr__(self):
        return f"{self.__class__.__name__}({', '.join(list(map(lambda k: f'{k}={self.__getattribute__(k)}', self.__dict__)))})"

    def __str__(self):
        return repr(self)


class Options:
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


class StorageOptions(Options, ReprMixin):
    """Options regarding storage of inputs and outputs."""

    def __init__(self, base_path, output_folder='scene3d', overwrite_ok=False):
        """
        :param base_path: Path to the folder containing the RGB and depth image folders.'
        :param output_folder: Name of the folder to save the results to (will be inside the folder `base_path`).
        :param overwrite_ok: Whether it is okay to replace old results.
        """
        self.base_path = base_path
        self.output_folder = output_folder
        self.overwrite_ok = overwrite_ok

    @staticmethod
    def add_args(parser: argparse.ArgumentParser):
        group = parser.add_argument_group('Storage Options')

        group.add_argument('--base_path', type=str,
                           help='Path to the folder containing the RGB and depth image folders.',
                           required=True)
        group.add_argument('--output_folder', type=str,
                           help='Name of the folder to save the results to (will be inside the folder `base_path`).',
                           default='scene3d')

        group.add_argument('--overwrite_ok', help='Whether it is okay to replace old results.',
                           action='store_true')

    @staticmethod
    def from_args(args) -> 'StorageOptions':
        return StorageOptions(
            base_path=args.base_path,
            output_folder=args.output_folder,
            overwrite_ok=args.overwrite_ok
        )


class DepthFormat(enum.Enum):
    DEPTH_TO_POINT = enum.auto()
    DEPTH_TO_PLANE = enum.auto()


class DepthOptions(Options, ReprMixin):
    """Options for depth maps."""

    def __init__(self, max_depth=10.0, dtype=np.uint16, depth_format=DepthFormat.DEPTH_TO_PLANE):
        """
        :param max_depth: The maximum depth value in the depth maps.
        :param dtype: The type of the depth values.
        :param depth_format: How depth values are measured in the depth maps.
        """
        assert dtype is np.uint8 or dtype is np.uint16, 'Only 8-bit and 16-bit depth maps are supported.'

        self.max_depth = max_depth
        self.depth_dtype = dtype
        self.depth_format = depth_format

    @staticmethod
    def add_args(parser: argparse.ArgumentParser):
        group = parser.add_argument_group('Depth Options')
        group.add_argument('--max_depth', type=float, help='The maximum depth value in the provided depth maps.',
                           default=10.0)
        group.add_argument('--depth_format', type=str, help='How depth values are measure in the depth maps.',
                           choices=['depth_to_point', 'depth_to_plane'], default='depth_to_plane')
        group.add_argument('--depth_dtype', type=str, help='The type of the depth values.', default='uint16',
                           choices=['uint8', 'uint16'])

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

        return DepthOptions(max_depth, dtype, depth_format)


class COLMAPOptions(Options, ReprMixin):
    quality_choices = ('low', 'medium', 'high', 'extreme')

    def __init__(self, is_single_camera=True, dense=False, quality='high', binary_path='/usr/local/bin/colmap'):
        self.binary_path = binary_path
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

    @staticmethod
    def from_args(args: argparse.Namespace) -> 'COLMAPOptions':
        return COLMAPOptions(
            binary_path=args.binary_path,
            is_single_camera=not args.multiple_cameras,
            dense=args.dense,
            quality=args.quality
        )


class MeshDecimationOptions(Options, ReprMixin):
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


class MaskDilationOptions(Options, ReprMixin):
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


class MeshFilteringOptions(Options, ReprMixin):
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
