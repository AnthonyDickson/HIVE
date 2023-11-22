"""This module contains the code for running the 'Fusion' family of 3D reconstruction algorithms (e.g. BundleFusion)."""

#  HIVE, creates 3D mesh videos.
#  Copyright (C) 2023 Anthony Dickson anthony.dickson9656@gmail.com
#
#  This program is free software: you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License
#  along with this program.  If not, see <https://www.gnu.org/licenses/>.

import logging
import numpy as np
import os
import re
import subprocess
import trimesh
from collections import OrderedDict
from os.path import join as pjoin
from tqdm import tqdm
from typing import Optional, List

from third_party.tsdf_fusion_python import fusion
from video2mesh.image_processing import dilate_mask
from video2mesh.io import VTMDataset
from video2mesh.options import BackgroundMeshOptions, MaskDilationOptions, MeshReconstructionMethod


def tsdf_fusion(dataset: VTMDataset, options=BackgroundMeshOptions(), num_frames=-1,
                frame_set: Optional[List[int]] = None) -> trimesh.Trimesh:
    """
    Run the TSDFFusion 3D reconstruction algorithm on a dataset (https://github.com/andyzeng/tsdf-fusion-python,
     http://3dmatch.cs.princeton.edu).

    :param dataset: The dataset to reconstruct the mesh from.
    :param options: The configuration for the voxel volume and depth map mask dilation.
    :param num_frames: (optional) Limits the number of frames used for the reconstruction.
        If set to -1, all frames from the dataset will be used.
    :param frame_set: (optional) Specify which frames to include in the reconstruction.
    :return: The reconstructed textured triangle mesh.
    """
    if num_frames == -1:
        num_frames = dataset.num_frames

    if frame_set is None:
        frame_set = range(num_frames)

    logging.info("Estimating voxel volume bounds...")
    vol_bnds = np.zeros((3, 2))

    # Dilate (increase size) of masks so that parts of the dynamic objects are not included in the final mesh
    # (this typically results in floating fragments in the static mesh.)
    mask_dilation_options = MaskDilationOptions(num_iterations=options.depth_mask_dilation_iterations)

    # Need to take inverse since poses are world-to-cam, but TSDFFusion appears to expect cam-to-world poses.
    camera_trajectory = dataset.camera_trajectory.inverse().to_homogenous_transforms()

    for i in tqdm(frame_set):
        # Read depth image and camera pose
        depth_im = dataset.bg_depth_dataset[i]
        cam_pose = camera_trajectory[i]  # 4x4 rigid transformation matrix

        # Compute camera view frustum and extend convex hull
        view_frust_pts = fusion.get_view_frustum(depth_im, dataset.camera_matrix, cam_pose)
        vol_bnds[:, 0] = np.minimum(vol_bnds[:, 0], np.amin(view_frust_pts, axis=1))
        vol_bnds[:, 1] = np.maximum(vol_bnds[:, 1], np.amax(view_frust_pts, axis=1))

    # ======================================================================================================== #
    # Integrate
    # ======================================================================================================== #
    voxel_count = np.ceil(np.product((vol_bnds[:, 1] - vol_bnds[:, 0]) / options.sdf_voxel_size))

    if options.sdf_max_voxels and voxel_count > options.sdf_max_voxels:
        voxel_size = (np.product(vol_bnds[:, 1] - vol_bnds[:, 0]) / options.sdf_max_voxels) ** (1 / 3)
        logging.info(f"Increasing voxel size to {voxel_size:.3f}: Using a voxel size of {options.sdf_voxel_size} would "
                     f"result in {voxel_count:,.0f} voxels, which is above the specified limit of "
                     f"{options.sdf_max_voxels:,d}.")
    else:
        voxel_size = options.sdf_voxel_size

    logging.info("Initializing voxel volume...")
    tsdf_vol = fusion.TSDFVolume(vol_bnds, voxel_size=voxel_size)

    logging.info("Fusing frames...")

    has_inpainted_frame_data = dataset.has_inpainted_frame_data

    for i in tqdm(frame_set):
        color_image = dataset.bg_rgb_dataset[i]
        depth_im = dataset.bg_depth_dataset[i]
        cam_pose = camera_trajectory[i]

        if not has_inpainted_frame_data:
            mask = dataset.mask_dataset[i]
            mask = dilate_mask(mask, mask_dilation_options)
            depth_im[mask > 0] = 0.0

        # Integrate observation into voxel volume (assume color aligned with depth)
        tsdf_vol.integrate(color_image, depth_im, dataset.camera_matrix, cam_pose, obs_weight=1.)

    # Get mesh from voxel volume and save to disk (can be viewed with Meshlab)
    verts, faces, norms, colors = tsdf_vol.get_mesh()

    # TODO: Cleanup mesh for floating fragments (e.g. via connected components analysis).
    # TODO: Fix this. It seems to mess up the order of the face vertices or something.
    # verts, faces = Video2Mesh.cleanup_with_connected_components(verts, faces, is_object=False, min_components=10)
    mesh = trimesh.Trimesh(vertices=verts, faces=faces, vertex_colors=colors, vertex_normals=norms)

    return mesh


class BundleFusionConfig:
    """Handles parsing and writing the configuration files used by BundleFusion."""

    def __init__(self, **kwargs):
        """
        :param kwargs: The key-value pairs configuration fields.
        """
        self.config_dict = OrderedDict(**kwargs)

    def __getitem__(self, key):
        return self.config_dict[key]

    def __setitem__(self, key, value):
        if key in self.config_dict and (value_type := type(value)) != (expected_type := type(self.config_dict[key])):
            logging.warning(f"The config file entry \"{key}\" is of type {expected_type} "
                            f"but it is being set to a new value of type {value_type}")

        self.config_dict[key] = value

    @staticmethod
    def load(f) -> 'BundleFusionConfig':
        """
        Load a configuration file from disk.
        :param f: The file path or object.
        :return: A BundleFusionConfig object.
        """
        if isinstance(f, str):
            with open(f, 'r') as fp:
                return BundleFusionConfig._read_file(fp)
        else:
            return BundleFusionConfig._read_file(f)

    @staticmethod
    def _read_file(fp) -> 'BundleFusionConfig':
        """
        Read and parse a BundleFusion configuration file from disk.

        :param fp: The file pointer to the configuration file (either the path or a file object).
        :return: The parsed configuration.
        """
        config = OrderedDict()

        # Lines in the configuration file are delimited with either a semicolon, or start with a hashtag for comments.
        delimiter_pattern = re.compile("[;#]|(//)")

        def convert_value(string_value):
            """
            Converts individual values from the raw config file into Python types.
            :param string_value: The configuration value straight from the file.
            :return: The configuration value as a Python friendly type.
            """
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

            # Drop the end of a line, or the entire line if it is just a comment.
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
        """
        Save the configuration to disk.
        :param f: The file path or object to write the configuration to.
        """
        if isinstance(f, str):
            with open(f, 'w') as fp:
                self._write_to_disk(fp)
        else:
            self._write_to_disk(f)

    def _write_to_disk(self, fp):
        """
        Convert the configuration to a BundleFusion friendly format and write to disk.

        :param fp: The file path or object to write to.
        """

        def convert_to_string(value) -> str:
            """
            Converts individual configuration values from Python types into the string format required by BundleFusion.

            :param value: The configuration value as a Python type.
            :return: The configuration value as a BundleFusion friendly string.
            """
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
                raise ValueError(f"The type '{type(value)}' is not supported for serialisation. "
                                 f"Supported types are list, float, int and str.")

        for attribute_name, value in self.config_dict.items():
            line = f"{attribute_name} = {convert_to_string(value)};\n"
            fp.write(line)


def get_bundle_fusion_path() -> str:
    try:
        bundle_fusion_path = os.environ['BUNDLE_FUSION_PATH']

        return bundle_fusion_path
    except KeyError:
        logging.error(f"Could not find the environment variable 'BUNDLE_FUSION_PATH'. "
                      f"Make sure that you set this environment variable or use the development Docker image "
                      f"(dican732/video2mesh:dev-cu116).")
        raise


def bundle_fusion(output_folder: str, dataset: VTMDataset,
                  options=BackgroundMeshOptions(MeshReconstructionMethod.BundleFusion), num_frames: int = -1) \
        -> trimesh.Trimesh:
    """
    Run the BundleFusion 3D reconstruction algorithm (http://graphics.stanford.edu/projects/bundlefusion/) on a dataset.

    :param output_folder: The name of the folder to save the results to (do not include the path to the folder).
    :param dataset: The dataset to reconstruct the mesh from.
    :param options: The configuration for the voxel volume and depth map mask dilation.
    :param num_frames: (optional) Limits the number of frames used for the reconstruction.
        If set to -1, all frames from the dataset will be used.
    :return: The reconstructed textured triangle mesh.
    """
    if num_frames == -1:
        num_frames = dataset.num_frames

    if dataset.has_inpainted_frame_data:
        # TODO: Make Bundle Fusion capable of using the inpainted frame data.
        logging.warning("The dataset has inpainted frame data available, "
                        "but Bundle Fusion is only set up to use the original frame data.")

    bundle_fusion_path = get_bundle_fusion_path()

    logging.info("Creating masked depth maps for BundleFusion...")
    dataset.create_masked_depth(MaskDilationOptions(num_iterations=options.depth_mask_dilation_iterations))
    dataset_path = os.path.abspath(dataset.base_path)
    bundle_fusion_output_path = pjoin(dataset_path, output_folder)
    os.makedirs(bundle_fusion_output_path, exist_ok=True)

    logging.info("Configuring BundleFusion...")
    default_config_path = pjoin(bundle_fusion_path, 'zParametersDefault.txt')
    config = BundleFusionConfig.load(default_config_path)
    config['s_SDFMaxIntegrationDistance'] = options.sdf_volume_size
    config['s_SDFVoxelSize'] = options.sdf_voxel_size
    config['s_cameraIntrinsicFx'] = int(dataset.fx)
    config['s_cameraIntrinsicFy'] = int(dataset.fy)
    config['s_cameraIntrinsicCx'] = int(dataset.cx)
    config['s_cameraIntrinsicCy'] = int(dataset.cy)
    config['s_generateMeshDir'] = bundle_fusion_output_path
    config_output_path = pjoin(bundle_fusion_output_path, 'bundleFusionConfig.txt')
    config.save(config_output_path)

    bundle_fusion_bin = os.environ['BUNDLE_FUSION_BIN']
    bundling_config_path = pjoin(bundle_fusion_path, 'zParametersBundlingDefault.txt')
    bundling_config = BundleFusionConfig.load(bundling_config_path)
    submap_size = bundling_config['s_submapSize']
    # the `+ submap_size` is to avoid 'off-by-one' like errors.
    bundling_config['s_maxNumImages'] = (num_frames + submap_size) // submap_size
    bundling_config_output_path = pjoin(bundle_fusion_output_path, 'bundleFusionBundlingConfig.txt')
    bundling_config.save(bundling_config_output_path)
    cmd = [bundle_fusion_bin, config_output_path, bundling_config_output_path,
           dataset_path, dataset.masked_depth_folder]
    log_path = pjoin(bundle_fusion_output_path, 'log.txt')
    logging.info(f"Running BundleFusion with command '{' '.join(cmd)}'")

    with subprocess.Popen(cmd, stdout=subprocess.PIPE, bufsize=1, universal_newlines=True) as p, \
            open(log_path, mode='w') as f, \
            tqdm(total=num_frames) as progress_bar:
        for line in p.stdout:
            f.write(line)

            if line.startswith("processing frame ") and progress_bar.n + 1 < num_frames:
                progress_bar.update()

    if p.returncode != 0:
        raise RuntimeError(f"BundleFusion returned a non-zero code, "
                           f"check the logs for what went wrong ({os.path.abspath(log_path)}).")

    # Read ply file into trimesh object
    mesh_path = pjoin(bundle_fusion_output_path, 'mesh.ply')

    with open(mesh_path, 'rb') as mesh_file:
        mesh = trimesh.load(mesh_file, file_type='ply')

    return mesh
