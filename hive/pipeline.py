"""
This module contains the code for running the pipeline end-to-end (minus the renderer).
"""

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

import argparse
import datetime
import json
import logging
import os
import shutil
import subprocess
import sys
import time
import traceback
from contextlib import contextmanager
from os.path import join as pjoin
from pathlib import Path
from typing import Optional, List, Tuple, Any, Union

import numpy as np
import openmesh as om
import resource
import torch
import trimesh
from PIL import Image
from scipy.spatial import Delaunay
from scipy.spatial.transform import Rotation
from tqdm import tqdm

from hive.dataset_adaptors import get_dataset
from hive.fusion import tsdf_fusion, bundle_fusion
from hive.geometric import point_cloud_from_depth, world2image, get_pose_components
from hive.image_processing import dilate_mask
from hive.io import HiveDataset, temporary_trajectory
from hive.options import StorageOptions, COLMAPOptions, MeshDecimationOptions, \
    MaskDilationOptions, MeshFilteringOptions, MeshReconstructionMethod, PipelineOptions, BackgroundMeshOptions, \
    ForegroundTrajectorySmoothingOptions, WebXROptions
from hive.pose_optimisation import ForegroundPoseOptimiser
from hive.utils import validate_camera_parameter_shapes, validate_shape, tqdm_imap, setup_logger, format_bytes, \
    set_key_path, timed_block, get_key_path


class Pipeline:
    """Converts a 2D video to a 3D video."""

    # This the folder the foreground and background meshes are written to.
    mesh_folder = "mesh"
    # This the folder the outputs from BundleFusion are written to.
    bundle_fusion_folder = "bundle_fusion"

    def __init__(self, options: PipelineOptions, storage_options: StorageOptions,
                 decimation_options=MeshDecimationOptions(),
                 dilation_options=MaskDilationOptions(), filtering_options=MeshFilteringOptions(),
                 colmap_options=COLMAPOptions(), static_mesh_options=BackgroundMeshOptions(),
                 webxr_options=WebXROptions(),
                 fts_options=ForegroundTrajectorySmoothingOptions()):
        """
        :param options: Options pertaining to the core program.
        :param storage_options: Options regarding storage of inputs and outputs.
        :param decimation_options: Options for mesh decimation.
        :param dilation_options: Options for mask dilation.
        :param filtering_options: Options for face filtering.
        :param colmap_options: Options for COLMAP.
        :param static_mesh_options: Options for creating the background static mesh.
        :param webxr_options: Options for configuring the WebXR renderer and metadata.
        :param fts_options: Options for foreground trajectory smoothing.
        """
        self.options = options
        self.storage_options = storage_options
        self.colmap_options = colmap_options
        self.decimation_options = decimation_options
        self.dilation_options = dilation_options
        self.filtering_options = filtering_options
        self.background_mesh_options = static_mesh_options
        self.webxr_options = webxr_options
        self.fts_options = fts_options

        self.profiling = dict()

        # TODO: Dump logs to output folder.
        setup_logger(self.options.log_file)

    @staticmethod
    def from_command_line() -> 'Pipeline':
        """
        Initialises an instance of the pipeline using command line arguments.

        :return: An instance of the pipeline.
        """
        parser = argparse.ArgumentParser("HIVE", description="Create 3D mesh videos from a RGB-D sequence with "
                                                             "camera trajectory annotations.")
        PipelineOptions.add_args(parser)
        StorageOptions.add_args(parser)
        MaskDilationOptions.add_args(parser)
        MeshFilteringOptions.add_args(parser)
        MeshDecimationOptions.add_args(parser)
        COLMAPOptions.add_args(parser)
        BackgroundMeshOptions.add_args(parser)
        WebXROptions.add_args(parser)

        args = parser.parse_args()

        pipeline_options = PipelineOptions.from_args(args)
        storage_options = StorageOptions.from_args(args)
        filtering_options = MeshFilteringOptions.from_args(args)
        dilation_options = MaskDilationOptions.from_args(args)
        decimation_options = MeshDecimationOptions.from_args(args)
        colmap_options = COLMAPOptions.from_args(args)
        static_mesh_options = BackgroundMeshOptions.from_args(args)
        webxr_options = WebXROptions.from_args(args)

        pipeline = Pipeline(
            options=pipeline_options,
            storage_options=storage_options,
            decimation_options=decimation_options,
            dilation_options=dilation_options,
            filtering_options=filtering_options,
            colmap_options=colmap_options,
            static_mesh_options=static_mesh_options,
            webxr_options=webxr_options
        )

        logging.debug(args)

        return pipeline

    @property
    def num_frames(self) -> int:
        return self.options.num_frames

    @property
    def estimate_pose(self) -> bool:
        return self.options.estimate_pose

    @property
    def estimate_depth(self) -> bool:
        return self.options.estimate_depth

    @property
    def mesh_path(self) -> str:
        """Where to save the foreground and background meshes to as a string path."""
        return pjoin(self.storage_options.output_path, self.mesh_folder)

    @contextmanager
    def timed_block(self, log_msg: Optional[str], key_path: list):
        """
        Log a message, run a block of code, and write the runtime of the block to `self.profiling`.

        :param log_msg: The optional message to log.
        :param key_path: The dictionary path(s) to write the runtime to, e.g. ['my_app', 'total_runtime'].
            Any nested dictionaries or keys that do not exist will be created automatically.
        """
        with timed_block(log_msg=log_msg, profiling=self.profiling, key_path=key_path) as timer:
            yield timer

    def run(self, dataset: Optional[HiveDataset] = None, compress=True):
        """
        Run the pipeline to convert a video or RGB-D dataset into a 3D video.

        :param dataset: By default, the pipeline will load the dataset specified in the command line options.
            You can specify a dataset here to use instead.
        :param compress: Whether to compress the output mesh files. Defaults to `True`.
        """
        start_time = time.time()
        self._reset_cuda_stats()

        with self.timed_block("Loading dataset...", ['timing', 'load_dataset', 'total']):
            if dataset is None:
                resize_to = None if self.options.disable_scaling else 640
                dataset = get_dataset(self.storage_options, self.colmap_options, self.options, resize_to=resize_to,
                                      profiling=self.profiling)

            if self.num_frames == -1:
                self.options.num_frames = dataset.num_frames
            else:
                # This handles the case where the specified number of frames is more than the total number of frames in
                # the non-truncated dataset.
                self.options.num_frames = min(self.num_frames, dataset.num_frames)

        with self.timed_block("Creating background mesh(es)...",
                              key_path=['timing', 'background_reconstruction', 'total']):
            background_scene = self._create_background_scene(dataset)

        with self.timed_block("Creating foreground mesh(es)...",
                              key_path=['timing', 'foreground_reconstruction', 'total']):
            foreground_scene = self._create_foreground_scene(dataset)

        with self.timed_block("Centering foreground and background scenes...", key_path=['timing', 'scene_centering']):
            foreground_scene, background_scene = self._center_scenes(dataset, foreground_scene, background_scene)

        with self.timed_block("Writing mesh data to disk...", key_path=['timing', 'mesh_export']):
            foreground_scene_path, background_scene_path = self._write_meshes_to_disk(
                mesh_path=self.mesh_path,
                foreground_scene=foreground_scene,
                background_scene=background_scene,
                overwrite_ok=self.storage_options.overwrite_ok
            )

        with self.timed_block("Compressing mesh data...", key_path=['timing', 'mesh_compression', 'total']):
            with self.timed_block(log_msg=None, key_path=['timing', 'mesh_compression', 'foreground']):
                if compress:
                    self._compress_with_draco(foreground_scene_path)

            with self.timed_block(log_msg=None, key_path=['timing', 'mesh_compression', 'background']):
                if compress:
                    self._compress_with_draco(background_scene_path)

        with self.timed_block(f"Exporting mesh data to local WebXR server folder {self.webxr_options.webxr_path}...",
                              key_path=['timing', 'webxr_export']):
            self._export_video_webxr(self.mesh_path, fg_scene_name="fg", bg_scene_name="bg",
                                     metadata=self._get_webxr_metadata(dataset),
                                     export_name=(self._get_dataset_name(dataset)))

        elapsed_time_seconds = time.time() - start_time

        self._print_summary(foreground_scene, background_scene,
                            foreground_scene_path, background_scene_path,
                            elapsed_time_seconds)

        self._write_profiling_data(path=pjoin(dataset.base_path, 'profiling.json'))

        logging.info(f"Start the WebXR server and go to this URL: "
                     f"{self.webxr_options.webxr_url}?video={self._get_dataset_name(dataset)}")

        if self.webxr_options.webxr_run_server:
            subprocess.run(["npm", "run", "start"], cwd=self.webxr_options.webxr_source_path)

    @staticmethod
    def _reset_cuda_stats():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.reset_accumulated_memory_stats()

    def _create_background_scene(self, dataset: HiveDataset) -> trimesh.Scene:
        """
        Create the background mesh(es) from an RGB-D dataset.

        :param dataset: An RGB-D dataset with known camera parameters.

        :return: The background scene.
        """
        if self.background_mesh_options.reconstruction_method == MeshReconstructionMethod.RGBD:
            background_scene = self._create_scene(dataset, num_frames=self.num_frames, include_background=True,
                                                  background_only=True)
        else:
            background_scene = self._create_empty_scene(dataset)

            frame_set = dataset.select_key_frames(threshold=self.background_mesh_options.key_frame_threshold)

            static_mesh = self.create_static_mesh(dataset, num_frames=self.num_frames,
                                                  options=self.background_mesh_options, frame_set=frame_set)

            # Convert colour to sRGB since that is what the renderer expects.
            # This is only needed for the TSDF Fusion meshes since they use vertex colours, for which THREE does not
            # automatically adjust the colour space.
            # The other meshing techniques, mainly the RGB-D approach in create_scene(...), use textures, and since we
            # export the mesh in the glTF format, the glTF loader in the renderer will automatically convert the
            # texture colour space to sRGB.
            vertex_colors = static_mesh.visual.vertex_colors[:, :3]
            static_mesh.visual.vertex_colors[:, :3] = (255 * np.power(vertex_colors / 255, 2.2)).astype(int)

            background_scene.add_geometry(static_mesh, node_name="000000")

        return background_scene

    def _create_foreground_scene(self, dataset: HiveDataset) -> trimesh.Scene:
        """
        Create the foreground mesh(es) from an RGB-D dataset.

        :param dataset: An RGB-D dataset with known camera parameters.

        :return: The foreground scene.
        """
        if self.options.background_only:
            foreground_scene = self._create_empty_scene(dataset)
        elif self.fts_options.num_epochs > 0:
            smoothed_trajectory = ForegroundPoseOptimiser(dataset, learning_rate=self.fts_options.learning_rate,
                                                          num_epochs=self.fts_options.num_epochs).run()

            with temporary_trajectory(dataset, smoothed_trajectory):
                foreground_scene = self._create_scene(dataset, num_frames=self.num_frames)
        else:
            foreground_scene = self._create_scene(dataset, num_frames=self.num_frames)

        return foreground_scene

    def _create_scene(self, dataset: HiveDataset, num_frames: int, include_background=False,
                      background_only=False) -> trimesh.Scene:
        """
        Create a 'scene', a collection of 3D meshes, from each frame in an RGB-D dataset.

        :param dataset: The set of RGB frames and depth maps to use as input.
        :param num_frames: How many frames to process. If set to -1, all frames are processed.
        :param include_background: Whether to include the background mesh for each frame.
        :param background_only: Whether to exclude dynamic foreground objects.
        :return: The Trimesh scene object.
        """
        # TODO: Can the various bool flag arguments be combined into a single bit flag?
        if num_frames == -1:
            num_frames = dataset.num_frames
        else:
            num_frames = num_frames

        if background_only:
            rgb_dataset = dataset.bg_rgb_dataset
            depth_dataset = dataset.bg_depth_dataset
            mask_dataset = dataset.mask_dataset
        else:
            rgb_dataset = dataset.rgb_dataset
            depth_dataset = dataset.depth_dataset
            mask_dataset = dataset.mask_dataset

        camera_matrix = dataset.camera_matrix

        scene = self._create_empty_scene(dataset)
        homogeneous_transformations = dataset.camera_trajectory.to_homogenous_transforms()

        def process_frame(index):
            rgb = rgb_dataset[index]
            depth = depth_dataset[index]
            mask_encoded = mask_dataset[index]
            # noinspection PyShadowingNames
            pose = homogeneous_transformations[index]

            frame_vertices = np.zeros((0, 3))
            frame_faces = np.zeros((0, 3))

            uv_atlas = []
            texture_atlas = []

            vertex_count = 0

            # Construct 3D Point Cloud
            rgb = np.ascontiguousarray(rgb[:, :, :3])
            rotation, translation = get_pose_components(pose)

            mask_start_i = 0 if include_background else 1
            # noinspection PyArgumentList
            mask_end_i = 1 if background_only else mask_encoded.max() + 1

            for object_id in range(mask_start_i, mask_end_i):
                with self.timed_block(log_msg=None,
                                      key_path=['timing', 'foreground_reconstruction', 'binary_mask_creation', index,
                                                object_id]):
                    is_object = object_id > 0

                    if is_object:
                        mask = mask_encoded == object_id
                        mask = dilate_mask(mask, self.dilation_options)
                    elif not is_object and dataset.has_inpainted_frame_data:
                        mask = np.ones(mask, dtype=bool)

                    coverage_ratio = mask.mean()

                    if coverage_ratio < 0.01 and not self.options.disable_coverage_constraint:
                        # TODO: Make minimum coverage ratio configurable?
                        logging.debug(
                            f"Skipping object #{object_id} in frame {index + 1} due to insufficient coverage.")
                        continue

                with self.timed_block(log_msg=None,
                                      key_path=['timing', 'foreground_reconstruction', 'per_object_mesh', 'total',
                                                index, object_id]):
                    vertices = point_cloud_from_depth(depth, mask, camera_matrix, rotation, translation)

                    if len(vertices) < 9:
                        logging.debug(f"Skipping object #{object_id} in frame {index + 1} "
                                      f"due to insufficient number of vertices ({len(vertices)}).")
                        continue

                    valid_pixels = mask & (depth > 0.0)
                    v, u = valid_pixels.nonzero()
                    # Need to take transpose since stacking UV coordinates gives (2, N) shaped array to get the
                    # expected (N, 2) shape.
                    points2d = np.vstack((u, v)).T
                    masked_depth = depth[valid_pixels]

                    with self.timed_block(log_msg=None,
                                          key_path=['timing', 'foreground_reconstruction', 'per_object_mesh',
                                                    'face_triangulation', index, object_id]):
                        # TODO: Filter long stretched out bits of floor attached to peoples' feet.
                        faces = self._triangulate_faces(points2d)

                with self.timed_block(log_msg=None,
                                      key_path=['timing', 'foreground_reconstruction', 'face_filtering', index,
                                                object_id]):
                    faces = self._filter_faces(points2d, masked_depth, faces, self.filtering_options)

                    if len(faces) < 1:
                        logging.debug(f"Skipping object #{object_id} in frame {index + 1} "
                                      f"due to insufficient number of faces ({len(faces)}).")
                        continue

                with self.timed_block(log_msg=None,
                                      key_path=['timing', 'foreground_reconstruction', 'mesh_decimation', index,
                                                object_id]):
                    set_key_path(self.profiling, ['mesh_decimation', 'vertex_count', 'before', index, object_id],
                                 len(vertices))
                    set_key_path(self.profiling, ['mesh_decimation', 'face_count', 'before', index, object_id],
                                 len(faces))

                    vertices, faces = self._decimate_mesh(vertices, faces, is_object, self.decimation_options)

                    set_key_path(self.profiling, ['mesh_decimation', 'vertex_count', 'after', index, object_id],
                                 len(vertices))
                    set_key_path(self.profiling, ['mesh_decimation', 'face_count', 'after', index, object_id],
                                 len(faces))

                with self.timed_block(log_msg=None,
                                      key_path=['timing', 'foreground_reconstruction', 'floater_removal', index,
                                                object_id]):
                    vertices, faces = self._cleanup_with_connected_components(
                        vertices, faces, is_object,
                        min_components=self.filtering_options.min_num_components
                    )

                with self.timed_block(log_msg=None,
                                      key_path=['timing', 'foreground_reconstruction', 'billboard', index, object_id]):
                    if is_object and self.options.billboard:
                        camera_space_points = rotation @ (vertices.T + translation)
                        camera_space_points[2, :] = np.median(camera_space_points[2, :])
                        vertices = (rotation.T @ (camera_space_points - translation)).T

                        # TODO: Fix crash due to new vertices being projected outside original frame and causing the
                        #  _get_mesh_texture_and_uv method to return no mesh. This is because the bounds of the
                        #  projected 2D points contains negative coordinates. Need to filter out vertices to project
                        #  to invalid 2D coordinates.

                with self.timed_block(log_msg=None,
                                      key_path=['timing', 'foreground_reconstruction', 'texturing', index, object_id]):
                    texture, uv = self._get_mesh_texture_and_uv(vertices, rgb, camera_matrix, rotation, translation)
                    texture_atlas.append(texture)
                    uv_atlas.append(uv)

                    frame_vertices = np.vstack((frame_vertices, vertices))
                    frame_faces = np.vstack((frame_faces, faces + vertex_count))
                    # Vertex count must be updated afterwards.
                    vertex_count += len(vertices)

            with self.timed_block(log_msg=None,
                                  key_path=['timing', 'foreground_reconstruction', 'texture_atlas_packing', index]):
                if len(texture_atlas) == 0:
                    # noinspection PyShadowingNames
                    mesh = trimesh.Trimesh()
                    logging.debug(f"Mesh for frame #{index + 1} is empty!")
                else:
                    packed_textures, packed_uv = self._pack_textures(texture_atlas, uv_atlas, n_rows=1)

                    # noinspection PyUnresolvedReferences,PyShadowingNames
                    mesh = trimesh.Trimesh(
                        frame_vertices,
                        frame_faces,
                        visual=trimesh.visual.TextureVisuals(
                            uv=packed_uv,
                            material=trimesh.visual.material.PBRMaterial(
                                baseColorTexture=Image.fromarray(packed_textures.astype(np.uint8)),
                            )
                        )
                    )

            return mesh

        if background_only:
            frames = dataset.select_key_frames(threshold=self.background_mesh_options.key_frame_threshold)
        else:
            frames = range(num_frames)

        logging.info("Processing frame data...")
        meshes = tqdm_imap(process_frame, frames)

        for i, mesh in zip(frames, meshes):
            if not mesh.is_empty:
                scene.add_geometry(mesh, node_name=f"{i:06d}")

        return scene

    def process_frame(self, dataset: HiveDataset, index: int, background_only=False, include_background=False,
                      enable_cc_analysis=True) -> trimesh.Trimesh:
        """
        Process a single frame from a dataset.

        This is similar to the `._create_scene(...)` method, but it does not include profiling code.
        It is mainly intended to be used outside the main pipeline in experiments.

        :param dataset: An RGB-D dataset.
        :param index: The index for the frame to use.
        :param background_only: Whether to include only the background (i.e., ignore foreground elements).
        :param include_background: Whether to include the background in the mesh. If `False`, only dynamic foreground
            elements are included.
        :param enable_cc_analysis: Whether to filter out floaters with connected component analysis.
        :return: A textured triangle mesh.
        """
        if background_only:
            rgb_dataset = dataset.bg_rgb_dataset
            depth_dataset = dataset.bg_depth_dataset
            mask_dataset = dataset.mask_dataset
        else:
            rgb_dataset = dataset.rgb_dataset
            depth_dataset = dataset.depth_dataset
            mask_dataset = dataset.mask_dataset

        camera_matrix = dataset.camera_matrix

        rgb = rgb_dataset[index]
        depth = depth_dataset[index]
        mask_encoded = mask_dataset[index]
        # noinspection PyShadowingNames
        pose = dataset.camera_trajectory.to_homogenous_transforms()[index]

        frame_vertices = np.zeros((0, 3))
        frame_faces = np.zeros((0, 3))

        uv_atlas = []
        texture_atlas = []

        vertex_count = 0

        # Construct 3D Point Cloud
        rgb = np.ascontiguousarray(rgb[:, :, :3])
        rotation, translation = get_pose_components(pose)

        mask_start_i = 0 if include_background else 1
        # noinspection PyArgumentList
        mask_end_i = 1 if background_only else mask_encoded.max() + 1

        for object_id in range(mask_start_i, mask_end_i):
            is_object = object_id > 0

            if is_object:
                mask = mask_encoded == object_id
                mask = dilate_mask(mask, self.dilation_options)
            elif not is_object and dataset.has_inpainted_frame_data:
                mask = np.ones(mask, dtype=bool)

            coverage_ratio = mask.mean()

            if coverage_ratio < 0.01 and not self.options.disable_coverage_constraint:
                # TODO: Make minimum coverage ratio configurable?
                logging.debug(
                    f"Skipping object #{object_id} in frame {index + 1} due to insufficient coverage.")
                continue

            vertices = point_cloud_from_depth(depth, mask, camera_matrix, rotation, translation)

            if len(vertices) < 9:
                logging.debug(f"Skipping object #{object_id} in frame {index + 1} "
                              f"due to insufficient number of vertices ({len(vertices)}).")
                continue

            valid_pixels = mask & (depth > 0.0)
            v, u = valid_pixels.nonzero()
            # Need to take transpose since stacking UV coordinates gives (2, N) shaped array to get the
            # expected (N, 2) shape.
            points2d = np.vstack((u, v)).T
            masked_depth = depth[valid_pixels]

            # TODO: Filter long stretched out bits of floor attached to peoples' feet.
            faces = self._triangulate_faces(points2d)

            faces = self._filter_faces(points2d, masked_depth, faces, self.filtering_options)

            if len(faces) < 1:
                logging.debug(f"Skipping object #{object_id} in frame {index + 1} "
                              f"due to insufficient number of faces ({len(faces)}).")
                continue

            if enable_cc_analysis:
                vertices, faces = self._cleanup_with_connected_components(
                    vertices, faces, is_object,
                    min_components=self.filtering_options.min_num_components
                )

            if is_object and self.options.billboard:
                camera_space_points = rotation @ (vertices.T + translation)
                camera_space_points[2, :] = np.median(camera_space_points[2, :])
                vertices = (rotation.T @ (camera_space_points - translation)).T

                # TODO: Fix crash due to new vertices being projected outside original frame and causing the
                #  _get_mesh_texture_and_uv method to return no mesh. This is because the bounds of the
                #  projected 2D points contains negative coordinates. Need to filter out vertices to project
                #  to invalid 2D coordinates.

            texture, uv = self._get_mesh_texture_and_uv(vertices, rgb, camera_matrix, rotation, translation)
            texture_atlas.append(texture)
            uv_atlas.append(uv)

            frame_vertices = np.vstack((frame_vertices, vertices))
            frame_faces = np.vstack((frame_faces, faces + vertex_count))
            # Vertex count must be updated afterwards.
            vertex_count += len(vertices)

        if len(texture_atlas) == 0:
            # noinspection PyShadowingNames
            mesh = trimesh.Trimesh()
            logging.debug(f"Mesh for frame #{index + 1} is empty!")
        else:
            packed_textures, packed_uv = self._pack_textures(texture_atlas, uv_atlas, n_rows=1)

            # noinspection PyUnresolvedReferences,PyShadowingNames
            mesh = trimesh.Trimesh(
                frame_vertices,
                frame_faces,
                visual=trimesh.visual.TextureVisuals(
                    uv=packed_uv,
                    material=trimesh.visual.material.PBRMaterial(
                        baseColorTexture=Image.fromarray(packed_textures.astype(np.uint8)),
                    )
                )
            )

        return mesh

    @staticmethod
    def _create_empty_scene(dataset: HiveDataset) -> trimesh.Scene:
        """
        Create an empty trimesh scene initialised with a camera.

        :param dataset: The dataset that has the camera intrinsics.
        :return: an empty trimesh scene initialised with a camera.
        """
        return trimesh.scene.Scene(
            camera=trimesh.scene.Camera(
                resolution=(dataset.frame_width, dataset.frame_height),
                focal=(dataset.fx, dataset.fy)
            )
        )

    @staticmethod
    def _triangulate_faces(points: np.ndarray) -> np.ndarray:
        """
        Triangulate and get face indices from a set of 2D points.

        :param points: The Nx2 array of points.
        :return: A Nx3 array of faces (3 vertex indices).
        """
        validate_shape(points, 'points', expected_shape=(None, 2))

        tri = Delaunay(points)
        faces = tri.simplices
        faces = np.asarray(faces)

        # Need to reverse winding order to ensure culling works as expected.
        faces = faces[:, ::-1]

        return faces

    @staticmethod
    def _filter_faces(points2d, depth, faces, options: MeshFilteringOptions):
        """
        Filter faces that connect distance vertices.

        :param points2d: The (?, 2) points in image space.
        :param depth: The (?,) depth values of the given 2D points.
        :param faces: The (?, 3) face vertex indices.

        :return: A filtered view of the faces that satisfy the image space and depth constraints.
        """
        validate_shape(points2d, 'points2d', expected_shape=(None, 2))
        validate_shape(depth, 'depth', expected_shape=(points2d.shape[0],))
        validate_shape(faces, 'faces', expected_shape=(None, 3))

        pixel_distances = np.linalg.norm(points2d[faces[:, [0, 2, 0]]] - points2d[faces[:, [1, 1, 2]]], axis=-1)

        depth_proj = depth.reshape((*depth.shape, 1))
        depth_distances = np.linalg.norm(depth_proj[faces[:, [0, 2, 0]]] - depth_proj[faces[:, [1, 1, 2]]], axis=-1)

        valid_faces = np.alltrue((pixel_distances <= options.max_pixel_distance) &
                                 (depth_distances <= options.max_depth_distance), axis=1)

        faces = faces[valid_faces]

        return faces

    @staticmethod
    def _decimate_mesh(vertices, faces, is_object, options: MeshDecimationOptions):
        """
        Decimate (simplify) a mesh.

        :param vertices: The (?, 3) vertices of the mesh.
        :param faces: The (?, 3) face vertex indices of the mesh.
        :param is_object: Whether the mesh is of a foreground object, or the background.

        :return: A reduced set of vertices and faces.
        """
        validate_shape(vertices, 'vertices', expected_shape=(None, 3))
        validate_shape(faces, 'faces', expected_shape=(None, 3))

        if (is_object and options.num_faces_object == -1) or (options.num_faces_background == -1):
            return vertices, faces

        # Construct temporary mesh.
        mesh = om.PolyMesh()
        mesh.add_vertices(vertices)
        mesh.add_faces(faces)

        d = om.PolyMeshDecimater(mesh)
        mh = om.PolyMeshModQuadricHandle()

        # add modules
        d.add(mh)
        d.module(mh).set_max_err(options.max_error)

        # decimate
        d.initialize()
        num_faces = options.num_faces_object if is_object else options.num_faces_background
        d.decimate_to_faces(n_faces=num_faces)

        mesh.garbage_collection()

        vertices = mesh.points()
        vertices = np.asarray(vertices)

        faces = mesh.face_vertex_indices()
        faces = np.asarray(faces)

        return vertices, faces

    @staticmethod
    def _cleanup_with_connected_components(vertices, faces, is_object=True, min_components=5):
        """
        Cleanup a mesh through analysis of the connected components.
        This gets rid of most floating bits of mesh.

        :param vertices: The (?, 3) vertices of the mesh.
        :param faces: The (?, 3) face vertex indices of the mesh.
        :param is_object: Whether the mesh is for an object, or the background. This determines whether the largest cluster
         is chosen (is_object=True) or if cluster has at least `min_components` components (faces) in it (is_object=False).
        :param min_components: The minimum number of components a cluster must have to prevent being filtered out from the
        final mesh. Larger values will result in larger mesh fragments being culled.

        :return: The filtered mesh.
        """
        validate_shape(vertices, 'vertices', expected_shape=(None, 3))
        validate_shape(faces, 'faces', expected_shape=(None, 3))

        mesh = trimesh.Trimesh(vertices=vertices, faces=faces)

        # noinspection PyUnresolvedReferences
        connected_components = trimesh.graph.connected_components(mesh.face_adjacency, min_len=min_components)
        mask = np.zeros(len(mesh.faces), dtype=bool)

        if connected_components:
            if is_object:
                # filter vertices/faces based on result of the largest component
                largest_component_index = np.argmax([len(c) for c in connected_components])
                mask[connected_components[largest_component_index]] = True
            else:
                mask[np.concatenate(connected_components)] = True
        else:
            logging.debug(f"Mesh found with no connected components.")

        mesh.update_faces(mask)

        vertices = mesh.vertices
        faces = mesh.faces

        return vertices, faces

    @staticmethod
    def _get_mesh_texture_and_uv(vertices, image, camera_matrix, rotation=np.eye(3), translation=np.zeros((3, 1)),
                                 scale_factor=1.0):
        """
        Get the cropped texture and UV coordinates for a given set of vertices.

        :param vertices: The (?, 3) vertices of the mesh.
        :param image: The (?, ?, 3) image to use as the texture for the mesh.
        :param camera_matrix: The (3, 3) camera intrinsics matrix.
        :param rotation: The (3, 3) camera rotation matrix.
        :param translation: The (3, 1) camera translation column vector.
        :param scale_factor: An optional value that scales the 2D points.

        :return: The cropped texture and UV coordinates.
        """
        validate_shape(vertices, 'vertices', expected_shape=(None, 3))
        validate_shape(image, 'image', expected_shape=(None, None, 3))
        validate_camera_parameter_shapes(camera_matrix, rotation, translation)

        uv, _ = world2image(vertices, camera_matrix, rotation, translation, scale_factor)

        min_u, min_v = np.min(np.round(uv), axis=0).astype(int)
        max_u, max_v = np.max(np.round(uv), axis=0).astype(int) + 1

        texture = image[min_v:max_v, min_u:max_u, :].copy()
        uv -= np.min(np.round(uv), axis=0)

        return texture, uv

    @staticmethod
    def _pack_textures(textures_atlas, uvs_atlas, n_rows=1):
        """I don't understand exactly how this function works...
        ask the original authors of this code: https://github.com/krematas/soccerontable/issues/new"""
        n_columns = len(textures_atlas) // n_rows + 1
        row_images = []
        canvas_h, canvas_w = 0, 0

        for i in range(n_rows):
            max_h, total_w, total_col = 0, 0, 0
            for j in range(n_columns):
                if i * n_columns + j >= len(textures_atlas):
                    break

                total_col = j

                h, w = textures_atlas[i * n_columns + j].shape[:2]
                if h > max_h:
                    max_h = h
                total_w += w

            row_image = np.zeros((max_h, total_w, 3), dtype=np.uint8)
            moving_w = 0

            for j in range(total_col + 1):
                h, w = textures_atlas[i * n_columns + j].shape[:2]
                row_image[:h, moving_w:(moving_w + w), :] = textures_atlas[i * n_columns + j]
                uvs_atlas[i * n_columns + j][:, 0] += moving_w
                moving_w += w

            if row_image.shape[1] > canvas_w:
                canvas_w = row_image.shape[1]

            canvas_h += row_image.shape[0]
            row_images.append(row_image)

        atlas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
        moving_h = 0

        for i in range(n_rows):
            h, w = row_images[i].shape[:2]
            atlas[moving_h:(moving_h + h), :w, :] = row_images[i]

            for j in range(n_columns):
                if i * n_columns + j >= len(textures_atlas):
                    break

                uvs_atlas[i * n_columns + j][:, 1] += moving_h
            moving_h += h

        final_uvs = np.zeros((0, 2))

        for uv_atlas in uvs_atlas:
            final_uvs = np.vstack((final_uvs, uv_atlas))

        final_uvs[:, 0] /= canvas_w
        final_uvs[:, 1] = 1. - final_uvs[:, 1] / canvas_h

        return atlas, final_uvs

    @classmethod
    def create_static_mesh(cls, dataset: HiveDataset, num_frames=-1, options=BackgroundMeshOptions(),
                           frame_set: Optional[List[int]] = None) -> trimesh.Trimesh:
        """
        Create a mesh of the static elements from an RGB-D dataset.

        :param dataset: The dataset to create the mesh from.
        :param num_frames: The max number of frames to use from the dataset.
        :param options: The options/settings for creating the static mesh.
        :param frame_set: The subset of frames to use for reconstruction (only applies to TSDFFusion method).

        :return: The reconstructed 3D mesh of the scene.
        """
        if num_frames < 1:
            num_frames = dataset.num_frames

        if frame_set is not None and len(frame_set) < 1:
            raise RuntimeError(f"`frame_set`, if not set to `None`, must be a list with at least one element.")

        if options.reconstruction_method == MeshReconstructionMethod.BundleFusion:
            mesh = bundle_fusion(cls.bundle_fusion_folder, dataset, options, num_frames)
        elif options.reconstruction_method == MeshReconstructionMethod.TSDFFusion:
            mesh = tsdf_fusion(dataset, options, num_frames, frame_set=frame_set)
        else:
            raise RuntimeError(f"Unsupported mesh reconstruction method: {options.reconstruction_method}")

        return mesh

    @classmethod
    def _write_meshes_to_disk(cls, mesh_path: str, foreground_scene: trimesh.Scene, background_scene: trimesh.Scene,
                              overwrite_ok=False) -> Tuple[str, str]:
        """
        Save the mesh files to disk.

        :param mesh_path: The folder to save the meshes to.
        :param foreground_scene: The scene that contains the mesh data for the dynamic objects.
        :param background_scene: The scene that contains the mesh data for the static background.
        :param overwrite_ok: Whether it is okay to replace files in `mesh_export_path`.

        :return: A 2-tuple containing the full path to the foreground and background mesh.
        """
        os.makedirs(mesh_path, exist_ok=overwrite_ok)
        foreground_scene_path = cls._write_mesh_to_disk(mesh_path, scene_name="fg", scene=foreground_scene)
        background_scene_path = cls._write_mesh_to_disk(mesh_path, scene_name="bg", scene=background_scene)

        return foreground_scene_path, background_scene_path

    @classmethod
    def _write_mesh_to_disk(cls, base_folder: str, scene_name: str, scene: trimesh.Scene) -> str:
        """
        Write a scene to disk.

        :param base_folder: The folder to save the mesh to.
        :param scene_name: The name of the scene. Will be used for the filename.
        :param scene: The scene object to export.

        :return: The path to the exported scene.
        """
        output_path = pjoin(base_folder, f'{scene_name}.glb')
        trimesh.exchange.export.export_scene(scene, output_path)
        logging.info(f"Wrote mesh data to {output_path}")

        return output_path

    def _compress_with_draco(self, path_to_glb: str):
        """
        Compress a glTF mesh (.glb, binary format) with draco compression to reduce the file size.

        :param path_to_glb: The path a glTF mesh file.
        """
        src_path = Path(path_to_glb)
        tmp_path = Path(os.path.join(src_path.parent, f"{src_path.stem}_tmp{src_path.suffix}"))

        command = ['draco_transcoder', '-i', str(src_path), '-o', str(tmp_path)]

        with subprocess.Popen(command, stdout=subprocess.PIPE, bufsize=1, universal_newlines=True) as p:
            for line in p.stdout:
                logging.debug(line.rstrip('\n'))

        if (return_code := p.wait()) != 0:
            logging.warning(f"draco_transcoder exited with code {return_code}.")

        size_before = os.path.getsize(src_path)
        size_after = os.path.getsize(tmp_path)

        data_saving = 1 - size_after / size_before
        compression_ratio = size_before / size_after

        shutil.move(tmp_path, src_path)

        logging.info(f"Compressed {src_path} with draco successfully "
                     f"({format_bytes(size_before)} before compression, {format_bytes(size_after)} after compression, "
                     f"{data_saving * 100:.2f}% data saving, {compression_ratio:.2f}:1 compression ratio).")

        if src_path.stem == 'fg':
            name = 'foreground'
        elif src_path.stem == 'bg':
            name = 'background'
        else:
            name = src_path.stem

        set_key_path(self.profiling, ['mesh_compression', name], {
            'uncompressed_file_size': size_before,
            'compressed_file_size': size_after,
            'data_saving': data_saving,
            'compression_ratio': compression_ratio
        })

    def _center_scenes(self, dataset: HiveDataset, foreground_scene: trimesh.Scene, background_scene: trimesh.Scene) -> \
            Tuple[trimesh.Scene, trimesh.Scene]:
        """
        Center the scenes at the world origin and orient them so that the render is looking at the front of scene.

        :param dataset: If BundleFusion was used to reconstruct the background, this dataset is used to align the scenes.
        :param foreground_scene: The scene that contains the mesh data for the dynamic objects.
        :param background_scene: The scene that contains the mesh data for the static background.

        :return: The centered scenes.
        """
        foreground_scene = foreground_scene.copy()
        background_scene = background_scene.copy()

        if self.background_mesh_options.reconstruction_method == MeshReconstructionMethod.BundleFusion:
            background_scene = self._align_bundle_fusion_reconstruction(dataset, background_scene)

        # Flip the scenes the right way up.
        rotate_right_way_up = np.eye(4, dtype=np.float32)
        rotate_right_way_up[:3, :3] = Rotation.from_euler('xyz', [0, 0, 180], degrees=True).as_matrix()

        foreground_scene.apply_transform(rotate_right_way_up)
        background_scene.apply_transform(rotate_right_way_up)

        if self.options.align_scene:
            # Scenes where the recording device was held at an angle and estimated pose is used will not sit flat on the
            # ground plane, this step attempts to fix that.
            # noinspection PyUnresolvedReferences
            transform_to_origin, _ = trimesh.bounds.oriented_bounds(background_scene, angle_digits=1)
            # This transform will result in the mesh being rotated 90 degrees about the local x-axis and then the local z-axis.
            # This rotation undoes that last bit.
            rotation = np.eye(4)
            rotation[:3, :3] = Rotation.from_euler('xyz', [-90, 0, 90], degrees=True).as_matrix()
            transform_to_origin = rotation @ transform_to_origin
            foreground_scene.apply_transform(transform_to_origin)
            background_scene.apply_transform(transform_to_origin)

        # Then move the scenes so that they are centered on the world origin.
        scene_bounds = self._get_scene_bounds(foreground_scene, background_scene)
        scene_centroid = np.mean(scene_bounds, axis=0)

        offset_from_center = np.array([-scene_centroid[0], -scene_bounds[0, 1], -scene_bounds[0, 2]])

        translation = np.eye(4, dtype=np.float32)
        translation[:3, 3] = offset_from_center

        foreground_scene.apply_transform(translation)
        background_scene.apply_transform(translation)

        return foreground_scene, background_scene

    def _align_bundle_fusion_reconstruction(self, dataset: HiveDataset, scene: trimesh.Scene) -> trimesh.Scene:
        """
        BundleFusion outputs a mesh that has been transformed in multiple ways (e.g., mirror and rotation).
        This function attempts to undo these transformations and align the background mesh with the foreground mesh.

        :param dataset: The RGB-D dataset the background scene was created from.
        :param scene: The background scene created with BundleFusion and the specified RGB-D dataset.

        :return: The aligned background scene.
        """
        pcd_bounds = np.zeros((2, 3), dtype=float)
        i = 0

        homogeneous_transformations = dataset.camera_trajectory.to_homogenous_transforms()

        for depth_map, pose, mask_encoded in \
                tqdm(zip(dataset.depth_dataset, homogeneous_transformations, dataset.mask_dataset),
                     total=self.num_frames):
            if i >= self.num_frames:
                break
            else:
                i += 1

            binary_mask = mask_encoded == 0

            rotation, translation = get_pose_components(pose)

            points3d = point_cloud_from_depth(depth_map, binary_mask, dataset.camera_matrix, rotation, translation)

            # Expand bounds
            pcd_bounds[0] = np.min(np.vstack((pcd_bounds[0], points3d.min(axis=0))), axis=0)
            pcd_bounds[1] = np.max(np.vstack((pcd_bounds[1], points3d.max(axis=0))), axis=0)

        pcd_centroid = pcd_bounds.mean(axis=0)

        aligned_scene = scene.copy()

        mirror = np.eye(4)
        mirror[0, 0] = -1  # BundleFusion reconstruction is flipped horizontally, this flips it back.
        aligned_scene.apply_transform(mirror)

        transform = np.eye(4)
        transform[:3, :3] = Rotation.from_euler('xyz', [105., 0., -5.], degrees=True).as_matrix()
        transform[:3, 3] = scene.centroid - pcd_centroid

        aligned_scene.apply_transform(transform)
        # Needed to fix (vertical?) offset.
        aligned_scene.apply_translation([1.25, 2.0, 1.0])

        return aligned_scene

    @staticmethod
    def _get_scene_bounds(foreground_scene, background_scene):
        """
        Get the bounds of two scenes.

        :return: A (2, 3) array where the first row is the minimum x, y and z coordinates and the second row the maximum.
        """
        fg_bounds = foreground_scene.bounds
        bg_bounds = background_scene.bounds

        # This happens if there are no people detected in the video,
        # thus no meshes in the foreground scene meaning the bounds are undefined a.k.a `None`.
        if fg_bounds is None:
            return bg_bounds

        scene_bounds = np.vstack([
            np.min(np.vstack((fg_bounds[0], bg_bounds[0])), axis=0),
            np.max(np.vstack((fg_bounds[1], bg_bounds[1])), axis=0),
        ])

        return scene_bounds

    @staticmethod
    def _get_dataset_name(dataset: HiveDataset) -> str:
        """Get the `name` of a dataset (the folder name)."""
        return Path(dataset.base_path).name

    def _get_webxr_metadata(self, dataset: HiveDataset) -> dict:
        """
        Create the metadata for the WebXR export.

        :param dataset: The dataset with information on the framerate and focal length.
        :return: A JSON-encodable dictionary containing the fields: `fps`, `num_frames` and `use_vertex_colour_for_bg`.
        """
        return dict(
            fps=dataset.fps,
            fov_y=int(dataset.fov_y),
            num_frames=self.num_frames,
            use_vertex_colour_for_bg=self.background_mesh_options.reconstruction_method != MeshReconstructionMethod.RGBD,
            add_ground_plane=self.webxr_options.webxr_add_ground_plane,
            add_sky_box=self.webxr_options.webxr_add_sky_box
        )

    def _export_video_webxr(self, mesh_path: str, fg_scene_name: str, bg_scene_name: str, metadata: dict,
                            export_name: str) -> str:
        """
        Exports the mesh data for viewing in the local WebXR renderer.

        :param mesh_path: The folder where the meshes were written to.
        :param fg_scene_name: The filename of the foreground scene without the file extension.
        :param bg_scene_name: The filename of the background scene without the file extension.
        :param metadata: The JSON-encodable metadata (e.g., fps, number of frames) dictionary.
        :param export_name: The name of the folder to write the exported mesh files to.

        :return the export folder path.
        """
        webxr_output_path = pjoin(self.webxr_options.webxr_path, export_name)
        os.makedirs(webxr_output_path, exist_ok=self.storage_options.overwrite_ok)

        metadata_filename = 'metadata.json'
        metadata_path = pjoin(mesh_path, metadata_filename)

        with open(metadata_path, 'w') as f:
            json.dump(metadata, f)

        def export_file(filename):
            shutil.copy(pjoin(mesh_path, filename), pjoin(webxr_output_path, filename))

        export_file(metadata_filename)
        export_file(f"{fg_scene_name}.glb")
        export_file(f"{bg_scene_name}.glb")

        logging.info(f"Exported mesh data to: {webxr_output_path}")

        return webxr_output_path

    def _print_summary(self, foreground_scene: trimesh.Scene, background_scene: trimesh.Scene,
                       foreground_scene_path: str, background_scene_path: str,
                       elapsed_time_seconds: float):
        """
        Print a text summary to the console and logs detailing the processing time, mesh size and GPU memory usage.

        :param foreground_scene: The collection of foreground mesh(es).
        :param background_scene: The collection of background mesh(es).
        :param foreground_scene_path: The path to where the foreground scene was saved.
        :param background_scene_path: The path to where the background scene was saved.
        :param elapsed_time_seconds: How long the pipeline took to run.
        """

        def count_tris(scene: trimesh.Scene):
            total = 0
            num_frames = 0

            for node_name in scene.graph.nodes_geometry:
                # which geometry does this node refer to
                _, geometry_name = scene.graph[node_name]

                # get the actual potential mesh instance
                geometry = scene.geometry[geometry_name]

                if hasattr(geometry, 'triangles'):
                    total += len(geometry.triangles)
                    num_frames += 1

            return total, num_frames

        fg_num_tris, num_fg_frames = count_tris(foreground_scene)
        bg_num_tris, num_bg_frames = count_tris(background_scene)
        total_num_tris = fg_num_tris + bg_num_tris
        fg_num_tris_per_frame = fg_num_tris / num_fg_frames if num_fg_frames > 0 else 0
        bg_num_tris_per_frame = bg_num_tris / num_bg_frames
        num_tris_per_frame = fg_num_tris_per_frame + bg_num_tris_per_frame

        fg_file_size = os.path.getsize(foreground_scene_path)
        bg_file_size = os.path.getsize(background_scene_path)
        total_file_size = fg_file_size + bg_file_size

        fg_file_size_per_frame = fg_file_size // num_fg_frames if num_fg_frames > 0 else 0
        bg_file_size_per_frame = bg_file_size // num_bg_frames
        file_size_per_frame = fg_file_size_per_frame + bg_file_size_per_frame

        elapsed_time = datetime.timedelta(seconds=elapsed_time_seconds)
        elapsed_time_per_frame = datetime.timedelta(seconds=elapsed_time_seconds / self.num_frames)

        self.profiling['frame_count'] = {
            'total': self.num_frames,
            'foreground': num_fg_frames,
            'background': num_bg_frames
        }

        self.profiling['elapsed_time'] = {
            'total': elapsed_time.total_seconds(),
            'per_frame': elapsed_time_per_frame.total_seconds()
        }

        self.profiling['file_size'] = {
            'total': total_file_size,
            'per_frame': file_size_per_frame,
            'foreground': {
                'total': fg_file_size,
                'per_frame': fg_file_size_per_frame
            },
            'background': {
                'total': bg_file_size,
                'per_frame': bg_file_size_per_frame
            }
        }

        self.profiling['peak_vram_usage'] = {
            'allocated': torch.cuda.max_memory_allocated(),
            'reserved': torch.cuda.max_memory_reserved()
        }

        try:
            # resource.getrusage(...).ru_maxrss is only available on Linux
            # This value is in kilobytes, but the rest of the stats are in bytes, so we multiply by 1000 to get bytes.
            self.profiling['peak_ram_usage'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1000
        except OSError:
            logging.error(''.join(traceback.format_exception(sys.exc_info())))
            logging.error(
                '`resource.getrusage(resource.RUSAGE_SELF).ru_maxrss` cannot be run on the current operation system.')
            self.profiling['peak_ram_usage'] = 0

        self.profiling['mesh_tri_count'] = {
            'total': total_num_tris,
            'per_frame': num_tris_per_frame,
            'foreground': {
                'total': fg_num_tris,
                'per_frame': fg_num_tris_per_frame
            },
            'background': {
                'total': bg_num_tris,
                'per_frame': bg_num_tris_per_frame
            }
        }

        logging.info('#' + '=' * 78 + '#')
        logging.info('#' + ' ' * 36 + 'Summary' + ' ' * 35 + '#')
        logging.info('#' + '=' * 78 + '#')
        logging.info(
            f"Processed {self.num_frames} frames ({num_fg_frames} fg, {num_bg_frames} bg) in {elapsed_time} ({elapsed_time_per_frame} per frame).")
        logging.info(
            f"    Total mesh triangles: {total_num_tris:>9,d} ({num_tris_per_frame:,.1f} per frame)")
        logging.info(f"        Foreground mesh: {fg_num_tris:>9,d} ({fg_num_tris_per_frame:,.1f} per frame)")
        logging.info(f"        Background mesh: {bg_num_tris:>9,d} ({bg_num_tris_per_frame:,.1f} per frame)")
        logging.info(
            f"    Total mesh size on disk: {format_bytes(total_file_size)} ({format_bytes(file_size_per_frame)} per frame)")
        logging.info(
            f"        Foreground Mesh: {format_bytes(fg_file_size)} ({format_bytes(fg_file_size_per_frame)} per frame)")
        logging.info(
            f"        Background Mesh: {format_bytes(bg_file_size)} ({format_bytes(bg_file_size_per_frame)} per frame)")

        logging.info(
            f"Peak GPU Memory Usage (Allocated): {format_bytes(torch.cuda.max_memory_allocated())} ({torch.cuda.max_memory_allocated():,d} Bytes)")
        logging.info(
            f"Peak GPU Memory Usage (Reserved): {format_bytes(torch.cuda.max_memory_reserved())} ({torch.cuda.max_memory_reserved():,d} Bytes)")

    def _write_profiling_data(self, path: str):
        profiling = self._calculate_profiling_statistics(self.profiling)

        with open(path, 'w') as f:
            json.dump(profiling, f)

    def _calculate_profiling_statistics(self, profiling: dict):
        key_paths = [
            ['timing', 'foreground_reconstruction', 'binary_mask_creation'],
            ['timing', 'foreground_reconstruction', 'per_object_mesh', 'total'],
            ['timing', 'foreground_reconstruction', 'per_object_mesh', 'face_triangulation'],
            ['timing', 'foreground_reconstruction', 'face_filtering'],
            ['timing', 'foreground_reconstruction', 'mesh_decimation'],
            ['timing', 'foreground_reconstruction', 'floater_removal'],
            ['timing', 'foreground_reconstruction', 'billboard'],
            ['timing', 'foreground_reconstruction', 'texturing'],
            ['timing', 'foreground_reconstruction', 'texture_atlas_packing'],
            ['mesh_decimation', 'vertex_count', 'before'],
            ['mesh_decimation', 'vertex_count', 'after'],
            ['mesh_decimation', 'face_count', 'before'],
            ['mesh_decimation', 'face_count', 'after']
        ]

        result = profiling.copy()

        for key_path in key_paths:
            try:
                dict_entry = get_key_path(result, key_path)
                count, total = self._traverse_dictionary(dict_entry)

                set_key_path(result, key_path, {
                    'count': count,
                    'total': total,
                    'mean': total / count if count > 0 else 0.0
                })

            except KeyError:
                logging.warning(traceback.format_exc())

        return result

    def _traverse_dictionary(self, dictionary: Union[dict, Any], count: int = 0, total: int = 0):
        if isinstance(dictionary, (float, int)):
            return 1, dictionary

        if not isinstance(dictionary, dict):
            return count, total

        for key in dictionary:
            sub_count, sub_total = self._traverse_dictionary(dictionary[key])
            count += sub_count
            total += sub_total

        return count, total


def main():
    program = Pipeline.from_command_line()
    program.run()


if __name__ == '__main__':
    main()
