import argparse
import enum
import numpy as np
import os


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

    def __init__(self, base_path, colour_folder='colour', depth_folder='depth', mask_folder='mask',
                 colmap_folder='colmap', output_folder='scene3d', overwrite_ok=False):
        """
        :param base_path: Path to the folder containing the RGB and depth image folders.'
        :param colour_folder: Name of the folder that contains the RGB images inside the folder `base_path`.'
        :param depth_folder: Name of the folder that contains the depth maps inside the folder `base_path`.'
        :param mask_folder: Name of the folder that contains the dynamic object masks inside the folder `base_path`.'
        :param colmap_folder: Name of the folder inside the folder `base_path` that contains the COLMAP output.
        :param output_folder: Name of the folder to save the results to (will be inside the folder `base_path`).
        :param overwrite_ok: Whether it is okay to replace old results.
        """
        self.base_path = base_path
        self.colour_folder = colour_folder
        self.depth_folder = depth_folder
        self.mask_folder = mask_folder
        self.colmap_folder = colmap_folder
        self.output_folder = output_folder
        self.overwrite_ok = overwrite_ok

    @property
    def colour_path(self):
        return os.path.join(self.base_path, self.colour_folder)

    @property
    def depth_path(self):
        return os.path.join(self.base_path, self.depth_folder)

    @property
    def mask_path(self):
        return os.path.join(self.base_path, self.mask_folder)

    @property
    def colmap_path(self):
        return os.path.join(self.base_path, self.colmap_folder)

    @staticmethod
    def add_args(parser: argparse.ArgumentParser):
        group = parser.add_argument_group('Storage Options')

        group.add_argument('--base_path', type=str,
                           help='Path to the folder containing the RGB and depth image folders.',
                           required=True)
        group.add_argument('--colour_folder', type=str,
                           help='Name of the folder that contains the RGB images inside the folder `base_path`.',
                           default='colour')
        group.add_argument('--depth_folder', type=str,
                           help='Name of the folder that contains the depth maps inside the folder `base_path`.',
                           default='depth')
        group.add_argument('--mask_folder', type=str,
                           help='Name of the folder that contains the dynamic object masks inside the folder `base_path`.',
                           default='mask')
        group.add_argument('--output_folder', type=str,
                           help='Name of the folder to save the results to (will be inside the folder `base_path`).',
                           default='scene3d')

        group.add_argument('--overwrite_ok', help='Whether it is okay to replace old results.',
                           action='store_true')

    @staticmethod
    def from_args(args) -> 'StorageOptions':
        return StorageOptions(
            base_path=args.base_path,
            colour_folder=args.colour_folder,
            depth_folder=args.depth_folder,
            mask_folder=args.mask_folder,
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
