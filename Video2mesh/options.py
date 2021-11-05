import argparse
import enum
import os

import numpy as np


class ReprMixin:
    """Mixin that provides a basic string representation for objects."""

    def __repr__(self):
        return f"{self.__class__.__name__}({', '.join(list(map(lambda k: f'{k}={self.__getattribute__(k)}', self.__dict__)))})"

    def __str__(self):
        return repr(self)


class StorageOptions(ReprMixin):
    """Options regarding storage of inputs and outputs."""

    def __init__(self, base_folder, colour_folder='colour', depth_folder='depth', mask_folder='mask',
                 output_folder='scene3d', overwrite_ok=False):
        """
        :param base_folder: Path to the folder containing the RGB and depth image folders.'
        :param colour_folder: Name of the folder that contains the RGB images inside the folder `base_folder`.'
        :param depth_folder: Name of the folder that contains the depth maps inside the folder `base_folder`.'
        :param mask_folder: Name of the folder that contains the dynamic object masks inside the folder `base_folder`.'
        :param output_folder: Name of the folder to save the results to (will be inside the folder `base_folder`).
        :param overwrite_ok: Whether it is okay to replace old results.
        """
        self.base_folder = base_folder
        self.colour_folder = os.path.join(base_folder, colour_folder)
        self.depth_folder = os.path.join(base_folder, depth_folder)
        self.mask_folder = os.path.join(base_folder, mask_folder)
        self.output_folder = output_folder
        self.overwrite_ok = overwrite_ok

    @staticmethod
    def add_args(parser: argparse.ArgumentParser):
        group = parser.add_argument_group('Storage Options')

        group.add_argument('--base_dir', type=str,
                           help='Path to the folder containing the RGB and depth image folders.',
                           required=True)
        group.add_argument('--colour_dir', type=str,
                           help='Name of the folder that contains the RGB images inside the folder `base_folder`.',
                           default='colour')
        group.add_argument('--depth_dir', type=str,
                           help='Name of the folder that contains the depth maps inside the folder `base_folder`.',
                           default='depth')
        group.add_argument('--mask_dir', type=str,
                           help='Name of the folder that contains the dynamic object masks inside the folder `base_folder`.',
                           default='mask')
        group.add_argument('--output_dir', type=str,
                           help='Name of the folder to save the results to (will be inside the folder `base_folder`).',
                           default='scene3d')

        group.add_argument('--overwrite_ok', help='Whether it is okay to replace old results.',
                           action='store_true')

    @staticmethod
    def from_args(args):
        return StorageOptions(
            base_folder=args.base_dir,
            colour_folder=args.colour_dir,
            depth_folder=args.depth_dir,
            mask_folder=args.mask_dir,
            output_folder=args.output_dir,
            overwrite_ok=args.overwrite_ok
        )


class DepthFormat(enum.Enum):
    DEPTH_TO_POINT = enum.auto()
    DEPTH_TO_PLANE = enum.auto()


class DepthOptions(ReprMixin):
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
    def from_args(args):
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
