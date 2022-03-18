import datetime
import subprocess

import argparse
import numpy as np
import openmesh as om
import os
import psutil
import shutil
import time
import trimesh
import warnings
from PIL import Image
from multiprocessing.pool import ThreadPool
from pathlib import Path
from scipy.spatial import Delaunay
from scipy.spatial.transform import Rotation
from trimesh.exchange.export import export_mesh
from trimesh.exchange.ply import load_ply

from Video2mesh.geometry import pose_vec2mat, point_cloud_from_depth, world2image, get_pose_components, dilate_mask, \
    point_cloud_from_rgbd
from Video2mesh.io import TUMAdaptor, StrayScannerAdaptor, VTMDataset, UnrealAdaptor, BundleFusionConfig, \
    COLMAPProcessor
from Video2mesh.options import StorageOptions, DepthOptions, COLMAPOptions, MeshDecimationOptions, \
    MaskDilationOptions, MeshFilteringOptions, MeshReconstructionMethod, Video2MeshOptions, StaticMeshOptions
from Video2mesh.utils import Timer, validate_camera_parameter_shapes, validate_shape, log
from thirdparty.tsdf_fusion_python import fusion

pjoin = os.path.join


class Video2Mesh:
    """Converts a 2D video to a 3D video."""

    def __init__(self, options: Video2MeshOptions, storage_options: StorageOptions,
                 decimation_options=MeshDecimationOptions(),
                 dilation_options=MaskDilationOptions(), filtering_options=MeshFilteringOptions(),
                 depth_options=DepthOptions(), colmap_options=COLMAPOptions(), static_mesh_options=StaticMeshOptions()):
        """
        :param options: Options pertaining to the core program.
        :param storage_options: Options regarding storage of inputs and outputs.
        :param decimation_options: Options for mesh decimation.
        :param dilation_options: Options for mask dilation.
        :param filtering_options: Options for face filtering.
        :param depth_options: Options for depth maps.
        :param colmap_options: Options for COLMAP.
        :param static_mesh_options: Options for creating the background static mesh.
        """
        self.options = options
        self.storage_options = storage_options
        self.mask_folder = storage_options
        self.depth_options = depth_options
        self.colmap_options = colmap_options
        self.decimation_options = decimation_options
        self.dilation_options = dilation_options
        self.filtering_options = filtering_options
        self.static_mesh_options = static_mesh_options

    @property
    def should_create_masks(self):
        return self.options.create_masks

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
    def estimate_depth(self):
        return self.options.estimate_depth

    @property
    def estimate_camera_params(self):
        return self.options.estimate_camera_params

    def run(self):
        timer = Timer()
        timer.start()

        storage_options = self.storage_options

        dataset = self._get_dataset()
        storage_options.base_path = dataset.base_path
        timer.split("configure dataset")

        foreground_output_folder = storage_options.output_folder
        background_output_folder = f"{foreground_output_folder}_bg"

        centering_transform = self._get_centering_transform(dataset)

        if self.include_background and self.static_background:
            background_scene = self.create_scene(dataset, timer,
                                                 include_background=True,
                                                 background_only=True)
            foreground_scene = self.create_scene(dataset, timer,
                                                 include_background=False,
                                                 background_only=False)

            foreground_scene.apply_transform(centering_transform)
            background_scene.apply_transform(centering_transform)

            self.write_results(dataset.base_path, foreground_output_folder, foreground_scene,
                               timer, storage_options.overwrite_ok)
            self.write_results(dataset.base_path, background_output_folder, background_scene, timer,
                               storage_options.overwrite_ok)
        else:
            fx, fy, height, width = self.extract_camera_params(dataset.camera_matrix)

            background_scene = trimesh.scene.Scene(
                camera=trimesh.scene.Camera(resolution=(width, height), focal=(fx, fy))
            )

            if not self.include_background:
                static_mesh = self.create_static_mesh(dataset, num_frames=self.num_frames,
                                                      options=self.static_mesh_options)
                background_scene.add_geometry(static_mesh)

                self.write_results(dataset.base_path, f"{background_output_folder}_unaligned", background_scene, timer,
                                   storage_options.overwrite_ok)

                # TODO: Change this so that the scene is transformed instead of the static mesh.
                # if self.static_mesh_options.reconstruction_method == MeshReconstructionMethod.BUNDLE_FUSION:
                #     static_mesh = self.align_bundle_fusion_reconstruction(dataset, static_mesh)
                #
                #     if self.estimate_camera_params and self.options.refine_colmap_poses:
                #         dataset.camera_trajectory = self._refine_colmap_poses()

                # TODO: Compress background mesh? PLY + Draco?
                timer.split("create static mesh")

                self.write_results(dataset.base_path, background_output_folder, background_scene, timer,
                                   storage_options.overwrite_ok)

            foreground_scene = self.create_scene(dataset, timer,
                                                 include_background=self.include_background,
                                                 background_only=False)

            self.write_results(dataset.base_path, f"{foreground_output_folder}_unaligned", foreground_scene, timer,
                               storage_options.overwrite_ok)

            foreground_scene.apply_transform(centering_transform)

            self.write_results(dataset.base_path, foreground_output_folder, foreground_scene, timer,
                               storage_options.overwrite_ok)

        self._export_video_webxr(dataset, background_output_folder)
        timer.stop()

        self._print_summary(foreground_scene, background_scene, foreground_output_folder, background_output_folder,
                            timer)

    def _print_summary(self, foreground_scene, background_scene, foreground_output_folder, background_output_folder,
                       timer):
        def get_folder_size(folder):
            total_size = os.path.getsize(folder)

            for item in os.listdir(folder):
                item_path = os.path.join(folder, item)

                if os.path.isfile(item_path):
                    total_size += os.path.getsize(item_path)
                elif os.path.isdir(item_path):
                    total_size += get_folder_size(item_path)

            return total_size

        def format_bytes(num):
            for unit in ["", "Ki", "Mi", "Gi", "Ti"]:
                if abs(num) < 1024.0:
                    return f"{num:3.1f} {unit}B"

                num /= 1024.0

            return f"{num:3.1f} PiB"

        def count_tris(scene: trimesh.Scene):
            total = 0

            for node_name in scene.graph.nodes_geometry:
                # which geometry does this node refer to
                _, geometry_name = scene.graph[node_name]

                # get the actual potential mesh instance
                geometry = scene.geometry[geometry_name]

                if hasattr(geometry, 'triangles'):
                    total += len(geometry.triangles)

            return total

        fg_file_size = get_folder_size(foreground_output_folder)
        bg_file_size = get_folder_size(background_output_folder)
        fg_file_size_per_frame = fg_file_size / self.num_frames
        bg_file_size_per_frame = bg_file_size / self.num_frames
        file_size_per_frame = fg_file_size_per_frame + bg_file_size_per_frame

        fg_num_tris = count_tris(foreground_scene)
        bg_num_tris = count_tris(background_scene)
        fg_num_tris_per_frame = fg_num_tris / self.num_frames
        bg_num_tris_per_frame = bg_num_tris / self.num_frames
        num_tris_per_frame = fg_num_tris_per_frame + bg_num_tris_per_frame

        elapsed_time_seconds = timer.stop_time - timer.start_time
        elapsed_time = datetime.timedelta(seconds=elapsed_time_seconds)
        elapsed_time_per_frame = datetime.timedelta(seconds=elapsed_time_seconds / self.num_frames)

        log('#' + '=' * 78 + '#')
        log('#' + ' ' * 36 + 'Summary' + ' ' * 35 + '#')
        log('#' + '=' * 78 + '#')
        log(f"Processed {self.num_frames} frames in {elapsed_time} ({elapsed_time_per_frame} per frame).")
        log(f"   Total mesh triangles: {fg_num_tris + bg_num_tris:>9,d} ({num_tris_per_frame:,.1f} per frame)")
        log(f"        Foreground mesh: {fg_num_tris:>9,d} ({fg_num_tris_per_frame:,.1f} per frame)")
        log(f"        Background mesh: {bg_num_tris:>9,d} ({num_tris_per_frame:,.1f} per frame)")
        log(f"Total mesh size on disk: {format_bytes(fg_file_size + bg_file_size)} ({format_bytes(file_size_per_frame)} per frame")
        log(f"     Dynamic Scene Mesh: {format_bytes(fg_file_size)} ({format_bytes(fg_file_size_per_frame)} per frame)")
        log(f"      Static Scene Mesh: {format_bytes(bg_file_size)} ({format_bytes(bg_file_size_per_frame)} per frame)")

    def _get_centering_transform(self, dataset):
        camera_trajectory = dataset.camera_trajectory
        pose = pose_vec2mat(camera_trajectory[0])
        R, t = get_pose_components(pose)

        rot_180 = Rotation.from_euler('xyz', [180, 180, 0], degrees=True).as_matrix()
        centering_transform = np.eye(4, dtype=np.float32)
        centering_transform[:3, :3] = rot_180 @ R.T
        centering_transform[:3, 3:] = -(R.T @ t)

        return centering_transform

    def _export_video_webxr(self, dataset, background_output_folder):
        storage_options = self.storage_options

        dataset_name = Path(dataset.base_path).name
        webxr_output_path = pjoin(self.options.webxr_path, dataset_name)
        os.makedirs(webxr_output_path, exist_ok=storage_options.overwrite_ok)

        def copy_video_files(video_folder):
            scene3d_path = pjoin(dataset.base_path, video_folder)
            scene3d_output_path = pjoin(webxr_output_path, video_folder)

            os.makedirs(scene3d_output_path, exist_ok=storage_options.overwrite_ok)
            shutil.copytree(scene3d_path, scene3d_output_path, dirs_exist_ok=storage_options.overwrite_ok)

        copy_video_files(storage_options.output_folder)
        copy_video_files(background_output_folder)

        log(f"Start the WebXR server and go to this URL: {self.options.webxr_url}?video={dataset_name}")

    def _get_dataset(self):
        dataset_path = self.storage_options.base_path

        if TUMAdaptor.is_valid_folder_structure(dataset_path):
            # TODO: Test TUMAdaptor
            dataset = TUMAdaptor(
                base_path=dataset_path,
                output_path=f"{dataset_path}_vtm",
                overwrite_ok=storage_options.overwrite_ok
            ).convert()
        elif StrayScannerAdaptor.is_valid_folder_structure(dataset_path):
            dataset = StrayScannerAdaptor(
                base_path=dataset_path,
                output_path=f"{dataset_path}_vtm",
                overwrite_ok=storage_options.overwrite_ok,
                resize_to=640,  # Resize the longest side to 640  # TODO: Make target image size configurable via cli.
                depth_confidence_filter_level=0  # TODO: Make depth confidence filter level configurable via cli.
            ).convert()
        elif UnrealAdaptor.is_valid_folder_structure(dataset_path):
            dataset = UnrealAdaptor(
                base_path=dataset_path,
                output_path=f"{dataset_path}_vtm",
                overwrite_ok=storage_options.overwrite_ok
            ).convert()
        elif VTMDataset.is_valid_folder_structure(dataset_path):
            dataset = VTMDataset(dataset_path, overwrite_ok=storage_options.overwrite_ok)
        elif not os.path.isdir(dataset_path):
            raise RuntimeError(f"Could open the path {dataset_path}.")
        else:
            raise RuntimeError(f"Could not recognise the dataset format for the dataset at {dataset_path}.")

        dataset.create_or_find_masks()

        if self.estimate_depth:
            dataset.use_estimated_depth(depth_options=self.depth_options)

        if self.estimate_camera_params:
            dataset.use_estimated_camera_parameters(colmap_options=colmap_options)

        self.options.num_frames = min(dataset.num_frames, self.num_frames)

        return dataset

    def _refine_colmap_poses(self):
        """
        Refine the pose data estimated with COLMAP with additional data from BundleFusion.

        :return: The refined camera trajectory data.
        """
        raise NotImplementedError

        refined_trajectory = np.eye(4)

        dataset_path = self.storage_options.base_path
        bundle_fusion_output_dir = pjoin(dataset_path, 'bundle_fusion')
        trajectory_path = pjoin(bundle_fusion_output_dir, 'trajectory.txt')
        bf_trajectory = np.loadtxt(trajectory_path)

        rows_with_data = np.any(np.isfinite(bf_trajectory), axis=1)
        # TODO: Scale COLMAP pose data with BundleFusion pose data. Could use sklearn's polyfit.

        bf_trajectory = bf_trajectory.reshape((-1, 4, 4))
        rot_as_quat = Rotation.from_matrix(bf_trajectory[:, :3, :3]).as_quat()

        # TODO: Interpolate missing poses

        return refined_trajectory

    def create_scene(self, dataset: VTMDataset, timer: Timer, include_background=False,
                     background_only=False) -> trimesh.Scene:
        """
        Create a 'scene', a collection of 3D meshes, from each frame in an RGB-D dataset.

        :param dataset: The set of RGB frames and depth maps to use as input.
        :param timer: The timer object for debug output.
        :param include_background: Whether to include the background mesh for each frame.
        :param background_only: Whether to exclude dynamic foreground objects.

        :return: The Trimesh scene object.
        """
        if background_only:
            num_frames = 1
        elif self.num_frames == -1:
            num_frames = dataset.num_frames
        else:
            num_frames = self.num_frames

        camera_matrix = dataset.camera_matrix

        fx, fy, height, width = self.extract_camera_params(camera_matrix)

        scene = trimesh.scene.Scene(
            camera=trimesh.scene.Camera(resolution=(width, height), focal=(fx, fy))
        )

        # TODO: Simplify progress logging and dump more detailed logs to disk.

        def process_frame(i):
            rgb = dataset.rgb_dataset[i]
            depth = dataset.depth_dataset[i]
            mask_encoded = dataset.mask_dataset[i]
            pose = dataset.camera_trajectory[i]

            timer.split(f"start mesh generation for frame {i:02d}")
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
                timer.split(f"\tMesh for object id #{object_id}")

                mask = mask_encoded == object_id

                is_object = object_id > 0

                coverage_ratio = mask.mean()

                if coverage_ratio < 0.01:
                    timer.split(f"\t\tSkipping object #{object_id} due to insufficient coverage.")
                    continue

                if is_object:
                    mask = dilate_mask(mask, self.dilation_options)
                    timer.split(f"\t\tdilate mask")

                vertices = point_cloud_from_depth(depth, mask, camera_matrix, R, t)
                timer.split("\t\tcreate point cloud")

                if len(vertices) < 9:
                    timer.split(
                        f"\t\tSkipping object #{object_id} due to insufficient number of vertices ({len(vertices)}).")
                    continue

                points2d, depth_proj = world2image(vertices, camera_matrix, R, t)
                timer.split("\t\tproject 3D points to pixel coordinates")

                faces = self.triangulate_faces(points2d)
                timer.split("\t\ttriangulate mesh")

                faces = self.filter_faces(points2d, depth_proj, faces, self.filtering_options)
                timer.split("\t\tfilter faces")

                vertices, faces = self.decimate_mesh(vertices, faces, is_object, self.decimation_options)
                timer.split("\t\tdecimate mesh")

                vertices, faces = self.cleanup_with_connected_components(
                    vertices, faces, is_object,
                    min_components=self.filtering_options.min_num_components
                )
                timer.split(f"\t\tCleanup mesh with connected component analysis")

                texture, uv = self.get_mesh_texture_and_uv(vertices, rgb, camera_matrix, R, t)
                texture_atlas.append(texture)
                uv_atlas.append(uv)
                timer.split("\t\tgenerate texture atlas and UVs")

                frame_vertices = np.vstack((frame_vertices, vertices))
                frame_faces = np.vstack((frame_faces, faces + vertex_count))
                # Vertex count must be updated afterwards.
                vertex_count += len(vertices)

                timer.split("\t\tadd object mesh to frame mesh")

            if len(texture_atlas) == 0:
                mesh = trimesh.Trimesh()
                warnings.warn(f"Mesh for frame #{i + 1} is empty!")
            else:
                packed_textures, packed_uv = self.pack_textures(texture_atlas, uv_atlas, n_rows=1)

                timer.split("\tpack texture atlas")

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

        pool = ThreadPool(processes=psutil.cpu_count(logical=False))
        meshes = pool.map(process_frame, range(num_frames))

        for i, mesh in enumerate(meshes):
            scene.add_geometry(mesh, node_name=f"{i:06d}")

        return scene

    @staticmethod
    def create_static_mesh(dataset: VTMDataset, num_frames=-1, options=StaticMeshOptions()):
        """
        Create a static mesh of the scene.

        :param dataset: The dataset to create the mesh from.
        :param num_frames: The max number of frames to use from the dataset.
        :param options: The options/settings for creating the static mesh.

        :return: The reconstructed 3D mesh of the scene.
        """
        if num_frames < 1:
            num_frames = dataset.num_frames

        if options.reconstruction_method == MeshReconstructionMethod.BUNDLE_FUSION:
            dataset.create_masked_depth(MaskDilationOptions(num_iterations=options.depth_mask_dilation_iterations))
            dataset_path = os.path.abspath(dataset.base_path)

            bundle_fusion_path = os.environ['BUNDLE_FUSION_PATH']
            default_config_path = pjoin(bundle_fusion_path, 'zParametersDefault.txt')
            config = BundleFusionConfig.load(default_config_path)
            config['s_SDFMaxIntegrationDistance'] = options.sdf_volume_size
            config['s_SDFVoxelSize'] = options.sdf_voxel_size
            config['s_cameraIntrinsicFx'] = int(dataset.fx)
            config['s_cameraIntrinsicFy'] = int(dataset.fy)
            config['s_cameraIntrinsicCx'] = int(dataset.cx)
            config['s_cameraIntrinsicCy'] = int(dataset.cy)

            bundle_fusion_output_dir = pjoin(dataset_path, 'bundle_fusion')
            config['s_generateMeshDir'] = bundle_fusion_output_dir

            config_output_path = pjoin(dataset_path, 'bundleFusionConfig.txt')
            config.save(config_output_path)

            bundle_fusion_bin = os.environ['BUNDLE_FUSION_BIN']
            bundling_config_path = pjoin(bundle_fusion_path, 'zParametersBundlingDefault.txt')
            bundling_config = BundleFusionConfig.load(bundling_config_path)

            submap_size = bundling_config['s_submapSize']
            bundling_config['s_maxNumImages'] = (
                                                        num_frames + submap_size) // submap_size  # the `+ submap_size` is to avoid 'off-by-one' like errors.

            bundling_config_output_path = pjoin(dataset_path, 'bundleFusionBundlingConfig.txt')
            bundling_config.save(bundling_config_output_path)

            return_code = subprocess.call([bundle_fusion_bin, config_output_path, bundling_config_output_path,
                                           dataset_path, dataset.masked_depth_folder])

            if return_code:
                raise RuntimeError(f"BundleFusion returned a non-zero code, check the logs for what went wrong.")

            # Read ply file into trimesh object
            mesh_path = pjoin(bundle_fusion_output_dir, 'mesh.ply')

            with open(mesh_path, 'rb') as mesh_file:
                mesh_data = load_ply(mesh_file)

            mesh = trimesh.Trimesh(**mesh_data)
        elif options.reconstruction_method == MeshReconstructionMethod.TSDF_FUSION:
            # ======================================================================================================== #
            # (Optional) This is an example of how to compute the 3D bounds
            # in world coordinates of the convex hull of all camera view
            # frustums in the dataset
            # ======================================================================================================== #
            log("Estimating voxel volume bounds...")

            cam_intr = dataset.camera_matrix
            vol_bnds = np.zeros((3, 2))

            # Dilate (increase size) of masks so that parts of the dynamic objects are not included in the final mesh
            # (this typically results in floating fragments in the static mesh.)
            mask_dilation_options = MaskDilationOptions(num_iterations=options.depth_mask_dilation_iterations)

            for i in range(num_frames):
                # Read depth image and camera pose
                mask = dataset.mask_dataset[i]
                mask = dilate_mask(mask, mask_dilation_options)
                depth_im = dataset.depth_dataset[i]
                depth_im[mask > 0] = 0.0
                cam_pose = pose_vec2mat(dataset.camera_trajectory[i])  # 4x4 rigid transformation matrix

                # Compute camera view frustum and extend convex hull
                view_frust_pts = fusion.get_view_frustum(depth_im, cam_intr, cam_pose)
                vol_bnds[:, 0] = np.minimum(vol_bnds[:, 0], np.amin(view_frust_pts, axis=1))
                vol_bnds[:, 1] = np.maximum(vol_bnds[:, 1], np.amax(view_frust_pts, axis=1))
            # ======================================================================================================== #

            # ======================================================================================================== #
            # Integrate
            # ======================================================================================================== #
            # Initialize voxel volume
            log("Initializing voxel volume...")
            tsdf_vol = fusion.TSDFVolume(vol_bnds, voxel_size=options.sdf_voxel_size)

            # Loop through RGB-D images and fuse them together
            t0_elapse = time.time()
            for i in range(num_frames):
                log("Fusing frame %d/%d" % (i + 1, (num_frames)))

                # Read RGB-D image and camera pose
                color_image = dataset.rgb_dataset[i]
                mask = dataset.mask_dataset[i]
                mask = dilate_mask(mask, mask_dilation_options)
                depth_im = dataset.depth_dataset[i]
                depth_im[mask > 0] = 0.0
                cam_pose = pose_vec2mat(dataset.camera_trajectory[i])

                # Integrate observation into voxel volume (assume color aligned with depth)
                tsdf_vol.integrate(color_image, depth_im, cam_intr, cam_pose, obs_weight=1.)

            fps = dataset.num_frames / (time.time() - t0_elapse)
            log("Average FPS: {:.2f}".format(fps))

            # Get mesh from voxel volume and save to disk (can be viewed with Meshlab)
            verts, faces, norms, colors = tsdf_vol.get_mesh()

            # TODO: Cleanup mesh for floating fragments (e.g. via connected components analysis).
            # TODO: Fix this. It seems to mess up the order of the face vertices or something.
            # verts, faces = Video2Mesh.cleanup_with_connected_components(verts, faces, is_object=False, min_components=10)

            mesh = trimesh.Trimesh(vertices=verts, faces=faces, vertex_colors=colors, vertex_normals=norms)
        else:
            raise RuntimeError(f"Unsupported mesh reconstruction method: {options.reconstruction_method}")

        return mesh

    @staticmethod
    def extract_camera_params(camera_intrinsics):
        cx = camera_intrinsics[0, 2]
        cy = camera_intrinsics[1, 2]
        width = int(2 * cx)
        height = int(2 * cy)
        fx = camera_intrinsics[0, 0]
        fy = camera_intrinsics[1, 1]

        return fx, fy, height, width

    @staticmethod
    def triangulate_faces(points):
        validate_shape(points, 'points', expected_shape=(None, 2))

        tri = Delaunay(points)
        faces = tri.simplices
        faces = np.asarray(faces)

        # Need to reverse winding order to ensure culling works as expected.
        faces = faces[:, ::-1]

        return faces

    @staticmethod
    def filter_faces(points2d, depth, faces, options: MeshFilteringOptions):
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
    def decimate_mesh(vertices, faces, is_object, options: MeshDecimationOptions):
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
    def cleanup_with_connected_components(vertices, faces, is_object=True, min_components=5):
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
    def get_mesh_texture_and_uv(vertices, image, K, R=np.eye(3), t=np.zeros((3, 1)), scale_factor=1.0):
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
    def pack_textures(textures_atlas, uvs_atlas, n_rows=1):
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
    def write_results(base_folder, output_dir, scene, timer, replace_old_results_ok=False):
        scene3d_path = pjoin(base_folder, output_dir)
        old_results_path = f"{scene3d_path}.old"

        if replace_old_results_ok:
            if os.path.exists(old_results_path):
                shutil.rmtree(old_results_path)

            if os.path.exists(scene3d_path):
                shutil.move(scene3d_path, old_results_path)

        try:
            os.makedirs(scene3d_path, exist_ok=False)
            output_files = trimesh.exchange.gltf.export_gltf(scene, merge_buffers=True)

            for filename in output_files:
                with open(pjoin(scene3d_path, filename), 'wb') as f:
                    f.write(output_files[filename])

            timer.split("write mesh data to disk")
        except:
            if replace_old_results_ok:
                if os.path.exists(scene3d_path):
                    shutil.rmtree(scene3d_path)

                if os.path.exists(old_results_path):
                    shutil.move(old_results_path, scene3d_path)
                timer.split("rolled back results after encountering fatal error")

            raise

    def align_bundle_fusion_reconstruction(self, dataset: VTMDataset, mesh: trimesh.Trimesh):
        # TODO: Check if the below transform is always the correct fix for reconstructions regardless of dataset.
        rgb = dataset.rgb_dataset[0][:, :, :3]
        depth_map = dataset.depth_dataset[0]
        pose = dataset.camera_trajectory[0]
        mask_encoded = dataset.mask_dataset[0]
        binary_mask = mask_encoded == 0

        pose_matrix = pose_vec2mat(pose)
        pose_matrix = np.linalg.inv(pose_matrix)
        R = pose_matrix[:3, :3]
        t = pose_matrix[:3, 3:]

        colours, points3d = point_cloud_from_rgbd(rgb=rgb, depth=depth_map, mask=binary_mask,
                                                  K=dataset.camera_matrix, R=R, t=t)
        points3d_homogenous = np.ones(shape=(points3d.shape[0], 4))
        points3d_homogenous[:, :3] = points3d

        centering_transform = self._get_centering_transform(dataset)
        points3d = (centering_transform @ points3d_homogenous.T).T[:, :3]

        rotation_transform = np.eye(4)
        rotation = Rotation.from_euler('xyz', [180, 180, 0], degrees=True).as_matrix()
        rotation_transform[:3, :3] = rotation

        translation_transform = np.eye(4)
        translation_transform[:3, 3] = -mesh.centroid

        scale_transform = np.eye(4)
        scale_transform[0, 0] = -1.

        aligned_mesh = mesh.copy()
        aligned_mesh = aligned_mesh.apply_transform(rotation_transform @ translation_transform @ scale_transform)

        sample_points = 2 ** 15  # TODO: Make number of mesh/pcd samples configurable via cli.
        bf_sampled_points, _ = trimesh.sample.sample_surface(mesh, count=sample_points)

        # TODO: Make number of ICP iterations and stopping threshold configurable via cli.
        transform, _, cost = trimesh.registration.icp(bf_sampled_points, points3d[sample_points::len(
            points3d) // sample_points], max_iterations=40, threshold=1e-10, scale=False, reflection=False)

        aligned_mesh = aligned_mesh.apply_transform(transform)

        return aligned_mesh


if __name__ == '__main__':
    parser = argparse.ArgumentParser("video2mesh.py", description="Create 3D meshes from a RGB-D sequence with "
                                                                  "camera trajectory annotations.")

    Video2MeshOptions.add_args(parser)
    StorageOptions.add_args(parser)
    DepthOptions.add_args(parser)
    MaskDilationOptions.add_args(parser)
    MeshFilteringOptions.add_args(parser)
    MeshDecimationOptions.add_args(parser)
    COLMAPOptions.add_args(parser)
    StaticMeshOptions.add_args(parser)

    args = parser.parse_args()
    print(args)

    video2mesh_options = Video2MeshOptions.from_args(args)
    storage_options = StorageOptions.from_args(args)
    depth_options = DepthOptions.from_args(args)
    filtering_options = MeshFilteringOptions.from_args(args)
    dilation_options = MaskDilationOptions.from_args(args)
    decimation_options = MeshDecimationOptions.from_args(args)
    colmap_options = COLMAPOptions.from_args(args)
    static_mesh_options = StaticMeshOptions.from_args(args)

    program = Video2Mesh(options=video2mesh_options,
                         storage_options=storage_options,
                         decimation_options=decimation_options,
                         dilation_options=dilation_options,
                         filtering_options=filtering_options,
                         depth_options=depth_options,
                         colmap_options=colmap_options,
                         static_mesh_options=static_mesh_options)
    program.run()
