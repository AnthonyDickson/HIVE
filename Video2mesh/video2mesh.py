from pathlib import Path

import argparse
import cv2
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
from scipy.spatial import Delaunay
from scipy.spatial.transform import Rotation
from trimesh.exchange.export import export_mesh
from typing import Optional

from Video2mesh.geometry import pose_vec2mat, point_cloud_from_depth, world2image, get_pose_components
from Video2mesh.io import TUMAdaptor, StrayScannerAdaptor, VTMDataset, UnrealAdaptor
from Video2mesh.options import StorageOptions, DepthOptions, COLMAPOptions, MeshDecimationOptions, \
    MaskDilationOptions, MeshFilteringOptions, Options
from Video2mesh.utils import Timer, validate_camera_parameter_shapes, validate_shape, log
from thirdparty.tsdf_fusion_python import fusion


class Video2MeshOptions(Options):

    def __init__(self, create_masks=False, include_background=False, static_background=False, num_frames=-1,
                 estimate_depth=False, estimate_camera_params=False,
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
        :param webxr_path: Where to export the 3D video files to.
        :param webxr_url: The URL to the WebXR 3D video player.
        """
        self.create_masks = create_masks
        self.include_background = include_background
        self.static_background = static_background
        self.num_frames = num_frames
        self.estimate_depth = estimate_depth
        self.estimate_camera_params = estimate_camera_params
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
            webxr_path=args.webxr_path,
            webxr_url=args.webxr_url
        )


class Video2Mesh:
    """Converts a 2D video to a 3D video."""

    def __init__(self, options: Video2MeshOptions, storage_options: StorageOptions,
                 decimation_options=MeshDecimationOptions(),
                 dilation_options=MaskDilationOptions(), filtering_options=MeshFilteringOptions(),
                 depth_options=DepthOptions(), colmap_options=COLMAPOptions()):
        """
        :param options: Options pertaining to the core program.
        :param storage_options: Options regarding storage of inputs and outputs.
        :param decimation_options: Options for mesh decimation.
        :param dilation_options: Options for mask dilation.
        :param filtering_options: Options for face filtering.
        :param depth_options: Options for depth maps.
        :param colmap_options: Options for COLMAP.
        """
        self.options = options
        self.storage_options = storage_options
        self.mask_folder = storage_options
        self.depth_options = depth_options
        self.colmap_options = colmap_options
        self.decimation_options = decimation_options
        self.dilation_options = dilation_options
        self.filtering_options = filtering_options

    @property
    def scale_factor(self):
        return self.options.scale_factor

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

        dataset = self._get_dataset(storage_options.base_path)
        storage_options.base_path = dataset.base_path

        timer.split("configure dataset")

        background_output_folder = f"{storage_options.output_folder}_bg"

        camera_trajectory = dataset.camera_trajectory

        pose = pose_vec2mat(camera_trajectory[0])
        R, t = get_pose_components(pose)
        rot_180 = Rotation.from_euler('xyz', [0, 0, 180], degrees=True).as_matrix()
        centering_transform = np.eye(4, dtype=np.float32)
        centering_transform[:3, :3] = rot_180 @ R.T
        centering_transform[:3, 3:] = -(R.T @ t)

        if self.include_background and self.static_background:
            background_scene = self.create_scene(dataset, timer,
                                                 include_background=True,
                                                 background_only=True)
            foreground_scene = self.create_scene(dataset, timer,
                                                 include_background=False,
                                                 background_only=False)

            foreground_scene.apply_transform(centering_transform)
            background_scene.apply_transform(centering_transform)

            self.write_results(dataset.base_path, storage_options.output_folder, foreground_scene,
                               timer, storage_options.overwrite_ok)
            self.write_results(dataset.base_path, background_output_folder, background_scene, timer,
                               storage_options.overwrite_ok)
        else:
            scene = self.create_scene(dataset, timer,
                                      include_background=self.include_background,
                                      background_only=False)

            scene.apply_transform(centering_transform)

            self.write_results(dataset.base_path, storage_options.output_folder, scene, timer,
                               storage_options.overwrite_ok)

            if not self.include_background:
                fx, fy, height, width = self.extract_camera_params(dataset.camera_matrix)

                bg_scene = trimesh.scene.Scene(
                    camera=trimesh.scene.Camera(resolution=(width, height), focal=(fx, fy))
                )

                static_mesh = self.create_static_mesh(dataset, num_frames=self.num_frames)
                timer.split("create static mesh")

                bg_scene.add_geometry(static_mesh)
                bg_scene.apply_transform(centering_transform)

                self.write_results(dataset.base_path, background_output_folder, bg_scene, timer,
                                   storage_options.overwrite_ok)

        self._export_video_webxr(dataset, background_output_folder)
        # TODO: Summarise results - how many frames? mesh size (on disk, vertices/faces per frame and total)
        timer.stop()

    def _export_video_webxr(self, dataset, background_output_folder):
        storage_options = self.storage_options

        dataset_name = Path(dataset.base_path).name
        webxr_output_path = os.path.join(self.options.webxr_path, dataset_name)
        os.makedirs(webxr_output_path, exist_ok=storage_options.overwrite_ok)

        def copy_video_files(video_folder):
            scene3d_path = os.path.join(dataset.base_path, video_folder)
            scene3d_output_path = os.path.join(webxr_output_path, video_folder)

            os.makedirs(scene3d_output_path, exist_ok=storage_options.overwrite_ok)
            shutil.copytree(scene3d_path, scene3d_output_path, dirs_exist_ok=storage_options.overwrite_ok)

        copy_video_files(storage_options.output_folder)
        copy_video_files(background_output_folder)

        log(f"Start the WebXR server and go to this URL: {self.options.webxr_url}?video={dataset_name}")

    def _get_dataset(self, dataset_path):
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
                resize_to=640,  # Resize the longest side to 640
                depth_confidence_filter_level=0
            ).convert()
        elif UnrealAdaptor.is_valid_folder_structure(dataset_path):
            dataset = UnrealAdaptor(
                base_path=dataset_path,
                output_path=f"{dataset_path}_vtm",
                overwrite_ok=storage_options.overwrite_ok
            ).convert()
        elif VTMDataset.is_valid_folder_structure(dataset_path):
            dataset = VTMDataset(dataset_path, overwrite_ok=storage_options.overwrite_ok)
        else:
            raise RuntimeError(f"Could not recognise the dataset format for the dataset at {dataset_path}.")

        dataset.create_or_find_masks()

        if self.estimate_depth:
            dataset.use_estimated_depth()

        if self.estimate_camera_params:
            dataset.use_estimated_camera_parameters(colmap_options=colmap_options)

        return dataset

    def create_scene(self, dataset: VTMDataset, timer: Timer, include_background=False,
                     background_only=False):
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
                    mask = self.dilate_mask(mask, self.dilation_options)
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
    def create_static_mesh(dataset: VTMDataset, num_frames=-1):
        """
        Create a static mesh of the scene.

        :param dataset: The dataset to create the mesh from.
        :param num_frames: The max number of frames to use from the dataset.

        :return: The reconstructed 3D mesh of the scene.
        """
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
        mask_dilation_options = MaskDilationOptions(num_iterations=10)

        if num_frames < 1:
            num_frames = dataset.num_frames

        for i in range(num_frames):
            # Read depth image and camera pose
            mask = dataset.mask_dataset[i]
            mask = Video2Mesh.dilate_mask(mask, mask_dilation_options)
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
        tsdf_vol = fusion.TSDFVolume(vol_bnds, voxel_size=0.02)

        # Loop through RGB-D images and fuse them together
        t0_elapse = time.time()
        for i in range(num_frames):
            log("Fusing frame %d/%d" % (i + 1, (num_frames)))

            # Read RGB-D image and camera pose
            color_image = dataset.rgb_dataset[i]
            mask = dataset.mask_dataset[i]
            mask = Video2Mesh.dilate_mask(mask, mask_dilation_options)
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
    def dilate_mask(mask, dilation_options: MaskDilationOptions):
        """
        Dilate an instance segmentation mask so that it covers a larger area.

        :param mask: The mask to enlarge/dilate.

        :return: The dilated mask.
        """
        validate_shape(mask, 'mask', expected_shape=(None, None))

        mask = mask.astype(np.float32)
        mask = cv2.dilate(mask.astype(float), dilation_options.filter, iterations=dilation_options.num_iterations)
        mask = mask.astype(bool)

        return mask

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
        scene3d_path = os.path.join(base_folder, output_dir)
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
                with open(os.path.join(scene3d_path, filename), 'wb') as f:
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

    args = parser.parse_args()
    print(args)

    video2mesh_options = Video2MeshOptions.from_args(args)
    storage_options = StorageOptions.from_args(args)
    depth_options = DepthOptions.from_args(args)
    filtering_options = MeshFilteringOptions.from_args(args)
    dilation_options = MaskDilationOptions.from_args(args)
    decimation_options = MeshDecimationOptions.from_args(args)
    colmap_options = COLMAPOptions.from_args(args)

    program = Video2Mesh(options=video2mesh_options,
                         storage_options=storage_options,
                         decimation_options=decimation_options,
                         dilation_options=dilation_options,
                         filtering_options=filtering_options,
                         depth_options=depth_options,
                         colmap_options=colmap_options)
    program.run()
