import argparse
import datetime
import json
import os
import shutil
import sys
import time
import warnings
from os.path import join as pjoin
from pathlib import Path
from typing import Optional, Set, List

import numpy as np
import openmesh as om
import trimesh
from PIL import Image
from numpy.polynomial import Polynomial
from scipy.spatial import Delaunay
from scipy.spatial.transform import Rotation
from tqdm import tqdm
from trimesh.exchange.export import export_mesh

from video2mesh.dataset_adaptors import TUMAdaptor, StrayScannerAdaptor, VideoAdaptor
from video2mesh.fusion import tsdf_fusion, bundle_fusion
from video2mesh.geometry import pose_vec2mat, point_cloud_from_depth, world2image, get_pose_components
from video2mesh.image_processing import dilate_mask
from video2mesh.io import VTMDataset, temporary_trajectory
from video2mesh.options import StorageOptions, COLMAPOptions, MeshDecimationOptions, \
    MaskDilationOptions, MeshFilteringOptions, MeshReconstructionMethod, PipelineOptions, StaticMeshOptions, \
    ForegroundTrajectorySmoothingOptions
from video2mesh.pose_optimisation import ForegroundPoseOptimiser
from video2mesh.utils import validate_camera_parameter_shapes, validate_shape, log, tqdm_imap


class Pipeline:
    """Converts a 2D video to a 3D video."""

    mesh_folder = "mesh"
    bundle_fusion_folder = "bundle_fusion"

    def __init__(self, options: PipelineOptions, storage_options: StorageOptions,
                 decimation_options=MeshDecimationOptions(),
                 dilation_options=MaskDilationOptions(), filtering_options=MeshFilteringOptions(),
                 colmap_options=COLMAPOptions(), static_mesh_options=StaticMeshOptions(),
                 fts_options=ForegroundTrajectorySmoothingOptions()):
        """
        :param options: Options pertaining to the core program.
        :param storage_options: Options regarding storage of inputs and outputs.
        :param decimation_options: Options for mesh decimation.
        :param dilation_options: Options for mask dilation.
        :param filtering_options: Options for face filtering.
        :param colmap_options: Options for COLMAP.
        :param static_mesh_options: Options for creating the background static mesh.
        :param fts_options: Options for foreground trajectory smoothing.
        """
        self.options = options
        self.storage_options = storage_options
        self.colmap_options = colmap_options
        self.decimation_options = decimation_options
        self.dilation_options = dilation_options
        self.filtering_options = filtering_options
        self.static_mesh_options = static_mesh_options
        self.fts_options = fts_options

    @property
    def num_frames(self):
        return self.options.num_frames

    @property
    def include_background(self):
        return self.options.include_background

    @property
    def static_background(self):
        return self.options.static_background

    @property
    def use_estimated_data(self):
        return self.options.use_estimated_data

    def run(self, dataset: Optional[VTMDataset] = None):
        start_time = time.time()

        storage_options = self.storage_options

        if dataset is None:
            dataset = self.get_dataset()

        # The root folder of the dataset may change if it had to be converted.
        storage_options.base_path = dataset.base_path
        log("Configured dataset")

        mesh_export_path = pjoin(dataset.base_path, self.mesh_folder)
        os.makedirs(mesh_export_path, exist_ok=storage_options.overwrite_ok)

        centering_transform = self._get_centering_transform(dataset)

        log("Creating background mesh(es)...")

        if self.include_background:
            background_scene = self._create_scene(dataset, include_background=True, background_only=True,
                                                  static_background=self.static_background)
        else:
            fx, fy, height, width = self._extract_camera_params(dataset.camera_matrix)

            background_scene = trimesh.scene.Scene(
                camera=trimesh.scene.Camera(resolution=(width, height), focal=(fx, fy))
            )

            if self.num_frames >= 1:
                if self.options.frame_step > 1:
                    frame_set = list(range(0, self.num_frames, self.options.frame_step))

                    if frame_set[-1] != self.num_frames - 1:
                        frame_set.append(self.num_frames - 1)
                else:
                    frame_set = list(range(self.num_frames))
            else:
                frame_set = None

            static_mesh = self._create_static_mesh(dataset, num_frames=self.num_frames,
                                                   options=self.static_mesh_options, frame_set=frame_set)
            background_scene.add_geometry(static_mesh, node_name="000000")

        self._write_results(mesh_export_path, scene_name=f"bg_unaligned", scene=background_scene)

        log("Creating foreground mesh(es)...")
        if self.fts_options.num_epochs > 0:
            smoothed_trajectory = ForegroundPoseOptimiser(dataset, learning_rate=self.fts_options.learning_rate,
                                                          num_epochs=self.fts_options.num_epochs).run()

            with temporary_trajectory(dataset, smoothed_trajectory):
                foreground_scene = self._create_scene(dataset)
        else:
            foreground_scene = self._create_scene(dataset)

        self._write_results(mesh_export_path, scene_name=f"fg_unaligned", scene=foreground_scene)

        log("Aligning foreground and background scenes...")
        foreground_scene.apply_transform(centering_transform)
        background_scene.apply_transform(centering_transform)

        if self.static_mesh_options.reconstruction_method == MeshReconstructionMethod.BundleFusion:
            background_scene = self._align_bundle_fusion_reconstruction(dataset, background_scene)

        scene_bounds = self._get_scene_bounds(foreground_scene, background_scene)
        scene_centroid = np.mean(scene_bounds, axis=0)

        offset_from_center = np.array([-scene_centroid[0], -scene_bounds[0, 1], -scene_bounds[0, 2]])
        foreground_scene.apply_translation(offset_from_center)
        background_scene.apply_translation(offset_from_center)

        foreground_scene_path = self._write_results(mesh_export_path, scene_name="fg", scene=foreground_scene)
        background_scene_path = self._write_results(mesh_export_path, scene_name="bg", scene=background_scene)

        elapsed_time_seconds = time.time() - start_time

        self._print_summary(foreground_scene, background_scene,
                            foreground_scene_path, background_scene_path,
                            elapsed_time_seconds)

        webxr_metadata = dict(
            fps=dataset.fps,
            use_vertex_colour_for_bg=not self.include_background
        )

        self._export_video_webxr(mesh_export_path, fg_scene_name="fg", bg_scene_name="bg",
                                 metadata=webxr_metadata, export_name=Path(dataset.base_path).name)

    @staticmethod
    def _get_scene_bounds(foreground_scene, background_scene):
        """
        Get the bounds of two scenes.

        :return: A (2, 3) array where the first row is the minimum x, y and z coordinates and the second row the maximum.
        """
        fg_bounds = foreground_scene.bounds
        bg_bounds = background_scene.bounds

        scene_bounds = np.vstack([
            np.min(np.vstack((fg_bounds[0], bg_bounds[0])), axis=0),
            np.max(np.vstack((fg_bounds[1], bg_bounds[1])), axis=0),
        ])

        return scene_bounds

    @staticmethod
    def _print_summary(foreground_scene: trimesh.Scene, background_scene: trimesh.Scene,
                       foreground_scene_path: str, background_scene_path: str,
                       elapsed_time_seconds: float):
        def format_bytes(num):
            for unit in ["", "Ki", "Mi", "Gi", "Ti"]:
                if abs(num) < 1024.0:
                    return f"{num:3.1f} {unit}B"

                num /= 1024.0

            return f"{num:3.1f} PiB"

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
        fg_num_tris_per_frame = fg_num_tris / num_fg_frames
        bg_num_tris_per_frame = bg_num_tris / num_bg_frames
        num_tris_per_frame = fg_num_tris_per_frame + bg_num_tris_per_frame

        num_frames = max(num_fg_frames, num_bg_frames)

        fg_file_size = os.path.getsize(foreground_scene_path)
        bg_file_size = os.path.getsize(background_scene_path)
        fg_file_size_per_frame = fg_file_size / num_fg_frames
        bg_file_size_per_frame = bg_file_size / num_bg_frames
        file_size_per_frame = fg_file_size_per_frame + bg_file_size_per_frame

        elapsed_time = datetime.timedelta(seconds=elapsed_time_seconds)
        elapsed_time_per_frame = datetime.timedelta(seconds=elapsed_time_seconds / num_frames)

        log('#' + '=' * 78 + '#')
        log('#' + ' ' * 36 + 'Summary' + ' ' * 35 + '#')
        log('#' + '=' * 78 + '#')
        log(f"Processed {num_frames} frames in {elapsed_time} ({elapsed_time_per_frame} per frame).")
        log(f"   Total mesh triangles: {fg_num_tris + bg_num_tris:>9,d} ({num_tris_per_frame:,.1f} per frame)")
        log(f"        Foreground mesh: {fg_num_tris:>9,d} ({fg_num_tris_per_frame:,.1f} per frame)")
        log(f"        Background mesh: {bg_num_tris:>9,d} ({bg_num_tris_per_frame:,.1f} per frame)")
        log(f"Total mesh size on disk: {format_bytes(fg_file_size + bg_file_size)} ({format_bytes(file_size_per_frame)} per frame)")
        log(f"     Dynamic Scene Mesh: {format_bytes(fg_file_size)} ({format_bytes(fg_file_size_per_frame)} per frame)")
        log(f"      Static Scene Mesh: {format_bytes(bg_file_size)} ({format_bytes(bg_file_size_per_frame)} per frame)")

    def _get_centering_transform(self, dataset):
        camera_trajectory = dataset.camera_trajectory
        pose = pose_vec2mat(camera_trajectory[0])
        R, t = get_pose_components(pose)

        rot_180 = Rotation.from_euler('xyz', [0, 0, 180], degrees=True).as_matrix()
        centering_transform = np.eye(4, dtype=np.float32)
        centering_transform[:3, :3] = rot_180 @ R.T
        centering_transform[:3, 3:] = -(R.T @ t)

        return centering_transform

    def _export_video_webxr(self, mesh_path: str, fg_scene_name: str, bg_scene_name: str, metadata: dict,
                            export_name: str):
        storage_options = self.storage_options

        webxr_output_path = pjoin(self.options.webxr_path, export_name)
        os.makedirs(webxr_output_path, exist_ok=storage_options.overwrite_ok)

        metadata_filename = 'metadata.json'
        metadata_path = pjoin(mesh_path, metadata_filename)

        with open(metadata_path, 'w') as f:
            json.dump(metadata, f)

        def export_file(filename):
            shutil.copy(pjoin(mesh_path, filename), pjoin(webxr_output_path, filename))

        export_file(metadata_filename)
        export_file(f"{fg_scene_name}.glb")
        export_file(f"{bg_scene_name}.glb")

        log(f"Start the WebXR server and go to this URL: {self.options.webxr_url}?video={export_name}")

    def get_dataset(self):
        storage_options = self.storage_options
        colmap_options = self.colmap_options

        dataset_path = storage_options.base_path

        if VTMDataset.is_valid_folder_structure(dataset_path):
            dataset = VTMDataset(dataset_path, overwrite_ok=storage_options.overwrite_ok)
        else:
            if TUMAdaptor.is_valid_folder_structure(dataset_path):
                dataset_converter = TUMAdaptor(
                    base_path=dataset_path,
                    output_path=f"{dataset_path}_vtm",
                    num_frames=self.options.num_frames,
                    frame_step=self.options.frame_step,
                    overwrite_ok=storage_options.overwrite_ok
                )
            elif StrayScannerAdaptor.is_valid_folder_structure(dataset_path):
                dataset_converter = StrayScannerAdaptor(
                    base_path=dataset_path,
                    output_path=f"{dataset_path}_vtm",
                    num_frames=self.options.num_frames,
                    frame_step=self.options.frame_step,
                    overwrite_ok=storage_options.overwrite_ok,
                    # Resize the longest side to 640  # TODO: Make target image size configurable via cli.
                    resize_to=640,
                    depth_confidence_filter_level=0  # TODO: Make depth confidence filter level configurable via cli.
                )
            elif VideoAdaptor.is_valid_folder_structure(dataset_path):
                path_no_extensions, _ = os.path.splitext(dataset_path)

                dataset_converter = VideoAdaptor(
                    base_path=dataset_path,
                    output_path=f"{path_no_extensions}_vtm",
                    num_frames=self.options.num_frames,
                    frame_step=self.options.frame_step,
                    overwrite_ok=storage_options.overwrite_ok,
                    resize_to=640
                )
            elif not os.path.isdir(dataset_path):
                raise RuntimeError(f"Could open the path {dataset_path} or it is not a folder.")
            else:
                raise RuntimeError(f"Could not recognise the dataset format for the dataset at {dataset_path}.")

            if self.use_estimated_data:
                dataset = dataset_converter.convert_from_rgb(colmap_options)
            else:
                dataset = dataset_converter.convert_from_ground_truth()

        self.options.num_frames = min(dataset.num_frames, self.num_frames)

        return dataset

    def _create_scene(self, dataset: VTMDataset, include_background=False, background_only=False,
                      static_background=False) -> trimesh.Scene:
        """
        Create a 'scene', a collection of 3D meshes, from each frame in an RGB-D dataset.

        :param dataset: The set of RGB frames and depth maps to use as input.
        :param include_background: Whether to include the background mesh for each frame.
        :param background_only: Whether to exclude dynamic foreground objects.
        :param static_background: Whether to only use the first frame for the background.

        :return: The Trimesh scene object.
        """
        if static_background:
            num_frames = 1
        elif self.num_frames == -1:
            num_frames = dataset.num_frames
        else:
            num_frames = self.num_frames

        camera_matrix = dataset.camera_matrix

        fx, fy, height, width = self._extract_camera_params(camera_matrix)

        scene = trimesh.scene.Scene(
            camera=trimesh.scene.Camera(resolution=(width, height), focal=(fx, fy))
        )

        def process_frame(i):
            rgb = dataset.rgb_dataset[i]
            depth = dataset.depth_dataset[i]
            mask_encoded = dataset.mask_dataset[i]
            pose = dataset.camera_trajectory[i]

            frame_vertices = np.zeros((0, 3))
            frame_faces = np.zeros((0, 3))

            uv_atlas = []
            texture_atlas = []

            vertex_count = 0

            # Construct 3D Point Cloud
            rgb = np.ascontiguousarray(rgb[:, :, :3])
            transform = pose_vec2mat(pose)
            transform = np.linalg.inv(transform)
            R = transform[:3, :3]
            t = transform[:3, 3:]

            mask_start_i = 0 if include_background else 1
            mask_end_i = 1 if background_only else mask_encoded.max() + 1

            for object_id in range(mask_start_i, mask_end_i):
                mask = mask_encoded == object_id

                is_object = object_id > 0

                coverage_ratio = mask.mean()

                if coverage_ratio < 0.01:
                    warnings.warn(f"Skipping object #{object_id} in frame {i + 1} due to insufficient coverage.")
                    continue

                if is_object:
                    mask = dilate_mask(mask, self.dilation_options)

                vertices = point_cloud_from_depth(depth, mask, camera_matrix, R, t)

                if len(vertices) < 9:
                    warnings.warn(f"Skipping object #{object_id} in frame {i + 1} "
                                  f"due to insufficient number of vertices ({len(vertices)}).")
                    continue

                # TODO: Filter long stretched out bits of floor attached to peoples' feet.
                points2d, depth_proj = world2image(vertices, camera_matrix, R, t)
                faces = self._triangulate_faces(points2d)
                faces = self._filter_faces(points2d, depth_proj, faces, self.filtering_options)
                vertices, faces = self._decimate_mesh(vertices, faces, is_object, self.decimation_options)

                vertices, faces = self._cleanup_with_connected_components(
                    vertices, faces, is_object,
                    min_components=self.filtering_options.min_num_components
                )

                texture, uv = self._get_mesh_texture_and_uv(vertices, rgb, camera_matrix, R, t)
                texture_atlas.append(texture)
                uv_atlas.append(uv)

                frame_vertices = np.vstack((frame_vertices, vertices))
                frame_faces = np.vstack((frame_faces, faces + vertex_count))
                # Vertex count must be updated afterwards.
                vertex_count += len(vertices)

            if len(texture_atlas) == 0:
                mesh = trimesh.Trimesh()
                warnings.warn(f"Mesh for frame #{i + 1} is empty!")
            else:
                packed_textures, packed_uv = self._pack_textures(texture_atlas, uv_atlas, n_rows=1)

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

        log("Processing frame data...")
        meshes = tqdm_imap(process_frame, range(num_frames))

        for i, mesh in enumerate(meshes):
            scene.add_geometry(mesh, node_name=f"{i:06d}")

        return scene

    @classmethod
    def _create_static_mesh(cls, dataset: VTMDataset, num_frames=-1, options=StaticMeshOptions(),
                            frame_set: Optional[List[int]] = None):
        """
        Create a static mesh of the scene.

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

    @staticmethod
    def _extract_camera_params(camera_intrinsics):
        cx = camera_intrinsics[0, 2]
        cy = camera_intrinsics[1, 2]
        width = int(2 * cx)
        height = int(2 * cy)
        fx = camera_intrinsics[0, 0]
        fy = camera_intrinsics[1, 1]

        return fx, fy, height, width

    @staticmethod
    def _triangulate_faces(points):
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
        num_vertices = options.num_vertices_object if is_object else options.num_vertices_background
        d.decimate_to(num_vertices)

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

        connected_components = trimesh.graph.connected_components(mesh.face_adjacency, min_len=min_components)
        mask = np.zeros(len(mesh.faces), dtype=bool)

        if connected_components:
            if is_object:
                # filter vertices/faces based on result of largest component
                largest_component_index = np.argmax([len(c) for c in connected_components])
                mask[connected_components[largest_component_index]] = True
            else:
                mask[np.concatenate(connected_components)] = True
        else:
            warnings.warn(f"Mesh found with no connected components.")

        mesh.update_faces(mask)

        vertices = mesh.vertices
        faces = mesh.faces

        return vertices, faces

    @staticmethod
    def _get_mesh_texture_and_uv(vertices, image, K, R=np.eye(3), t=np.zeros((3, 1)), scale_factor=1.0):
        """
        Get the cropped texture and UV coordinates for a given set of vertices.

        :param vertices: The (?, 3) vertices of the mesh.
        :param image: The (?, ?, 3) image to use as the texture for the mesh.
        :param K: The (3, 3) camera intrinsics matrix.
        :param R: The (3, 3) camera rotation matrix.
        :param t: The (3, 1) camera translation column vector.
        :param scale_factor: An optional value that scales the 2D points.

        :return: The cropped texture and UV coordinates.
        """
        validate_shape(vertices, 'vertices', expected_shape=(None, 3))
        validate_shape(image, 'image', expected_shape=(None, None, 3))
        validate_camera_parameter_shapes(K, R, t)

        uv, _ = world2image(vertices, K, R, t, scale_factor)

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

            row_image = np.zeros((max_h, total_w, 3), dtype=np.float32)
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

        atlas = np.zeros((canvas_h, canvas_w, 3), dtype=np.float32)
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

    @staticmethod
    def _write_results(base_folder, scene_name, scene) -> str:
        """
        Write a scene to disk.

        :param base_folder: The folder to save the mesh to.
        :param scene_name: The name of the scene. Will be used for the filename.
        :param scene: The scene object to export.

        :return: The path to the exported scene.
        """
        output_path = pjoin(base_folder, f'{scene_name}.glb')
        trimesh.exchange.export.export_scene(scene, output_path)
        log("Wrote mesh data to disk")

        return output_path

    def _align_bundle_fusion_reconstruction(self, dataset: VTMDataset, scene: trimesh.Scene):
        pcd_bounds = np.zeros((2, 3), dtype=float)
        i = 0

        for depth_map, pose, mask_encoded in \
                tqdm(zip(dataset.depth_dataset, dataset.camera_trajectory, dataset.mask_dataset),
                     total=self.num_frames):
            if i >= self.num_frames:
                break
            else:
                i += 1

            binary_mask = mask_encoded == 0

            pose_matrix = pose_vec2mat(pose)
            pose_matrix = np.linalg.inv(pose_matrix)
            R = pose_matrix[:3, :3]
            t = pose_matrix[:3, 3:]

            points3d = point_cloud_from_depth(depth_map, binary_mask, dataset.camera_matrix, R, t)

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


def main():
    parser = argparse.ArgumentParser("video2mesh.py", description="Create 3D meshes from a RGB-D sequence with "
                                                                  "camera trajectory annotations.")
    PipelineOptions.add_args(parser)
    StorageOptions.add_args(parser)
    MaskDilationOptions.add_args(parser)
    MeshFilteringOptions.add_args(parser)
    MeshDecimationOptions.add_args(parser)
    COLMAPOptions.add_args(parser)
    StaticMeshOptions.add_args(parser)

    args = parser.parse_args()
    log(args)

    video2mesh_options = PipelineOptions.from_args(args)
    storage_options = StorageOptions.from_args(args)
    filtering_options = MeshFilteringOptions.from_args(args)
    dilation_options = MaskDilationOptions.from_args(args)
    decimation_options = MeshDecimationOptions.from_args(args)
    colmap_options = COLMAPOptions.from_args(args)
    static_mesh_options = StaticMeshOptions.from_args(args)

    program = Pipeline(options=video2mesh_options,
                       storage_options=storage_options,
                       decimation_options=decimation_options,
                       dilation_options=dilation_options,
                       filtering_options=filtering_options,
                       colmap_options=colmap_options,
                       static_mesh_options=static_mesh_options)
    program.run()


if __name__ == '__main__':
    main()
