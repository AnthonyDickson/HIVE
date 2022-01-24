import argparse
import os
import shutil
import warnings
from multiprocessing.pool import ThreadPool
from typing import Optional

import cv2
import numpy as np
import openmesh as om
import psutil
import trimesh
from PIL import Image
from detectron2.utils.logger import setup_logger
from scipy.spatial import Delaunay

from Video2mesh.geometry import pose_vec2mat, point_cloud_from_depth, world2image
from Video2mesh.io import TUMAdaptor, StrayScannerAdaptor, VTMDataset
from Video2mesh.options import StorageOptions, ReprMixin, DepthOptions, COLMAPOptions, Options
from Video2mesh.utils import Timer, validate_camera_parameter_shapes, validate_shape

setup_logger()


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


class Video2Mesh:
    def __init__(self, storage_options, decimation_options=MeshDecimationOptions(),
                 dilation_options=MaskDilationOptions(), filtering_options=MeshFilteringOptions(),
                 depth_options=DepthOptions(), colmap_options: Optional[COLMAPOptions] = None,
                 should_create_masks=False, batch_size=8, num_frames=-1, fps=60, scale_factor=1.0,
                 include_background=False, static_background=False,
                 estimate_depth=False, estimate_camera_params=False):
        # TODO: Fill out Video2Mesh __init__(...) docstring.
        """
        :param storage_options:
        :param decimation_options:
        :param dilation_options:
        :param filtering_options:
        :param depth_options:
        :param colmap_options:
        :param should_create_masks:
        :param batch_size:
        :param num_frames:
        :param fps:
        :param scale_factor:
        :param include_background:
        :param static_background:
        :param estimate_depth:
        :param estimate_camera_params:
        """
        self.storage_options = storage_options
        self.mask_folder = storage_options
        self.depth_options = depth_options
        self.colmap_options = colmap_options
        self.decimation_options = decimation_options
        self.dilation_options = dilation_options
        self.filtering_options = filtering_options
        # TODO: Put loose params into `DatasetOptions` class.
        self.should_create_masks = should_create_masks
        self.fps = fps
        self.scale_factor = scale_factor
        self.batch_size = batch_size
        self.num_frames = num_frames
        self.include_background = include_background
        self.static_background = static_background
        self.estimate_depth = estimate_depth
        self.estimate_camera_params = estimate_camera_params

    def run(self):
        timer = Timer()
        timer.start()

        dataset_path = self.storage_options.base_path

        dataset = self._get_dataset(dataset_path)

        timer.split("configure dataset")

        camera_trajectory = dataset.camera_trajectory

        if self.include_background and self.static_background:
            background_output_folder = f"{self.storage_options.output_folder}_bg"
            background_scene = self.create_scene(dataset, timer,
                                                 include_background=True,
                                                 background_only=True)
            foreground_scene = self.create_scene(dataset, timer,
                                                 include_background=False,
                                                 background_only=False)

            T = np.eye(4)
            T[:3, 3] = -camera_trajectory[0][3:]
            T[:3, :3] = cv2.Rodrigues(camera_trajectory[0][:3])[0]
            foreground_scene.apply_transform(T)
            background_scene.apply_transform(T)

            self.write_results(dataset_path, self.storage_options.output_folder, foreground_scene,
                               timer, self.storage_options.overwrite_ok)
            self.write_results(dataset_path, background_output_folder, background_scene, timer,
                               self.storage_options.overwrite_ok)
        else:
            scene = self.create_scene(dataset, timer,
                                      include_background=self.include_background,
                                      background_only=False)
            T = np.eye(4)
            T[:3, 3] = -camera_trajectory[0][3:]
            T[:3, :3] = cv2.Rodrigues(camera_trajectory[0][:3])[0]

            scene.apply_transform(T)
            # TODO: Undo initial pose so that video is centered at world origin with no rotation.
            self.write_results(dataset_path, self.storage_options.output_folder, scene, timer,
                               self.storage_options.overwrite_ok)

        # TODO: Summarise results - how many frames? mesh size (on disk, vertices/faces per frame and total)
        timer.stop()

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

        fx, fy, height, width = self.extract_camera_params(dataset.camera_matrix)

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
                    timer.split(f"\t\terode mask")

                vertices = point_cloud_from_depth(depth, mask, K, R, t, self.scale_factor)
                timer.split("\t\tcreate point cloud")

                if len(vertices) < 9:
                    timer.split(
                        f"\t\tSkipping object #{object_id} due to insufficient number of vertices ({len(vertices)}).")
                    continue

                points2d, depth_proj = world2image(vertices, K, R, t, self.scale_factor)
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

                texture, uv = self.get_mesh_texture_and_uv(vertices, rgb, K, R, t, self.scale_factor)
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
        meshes = pool.starmap(process_frame, range(num_frames))
        for i, mesh in enumerate(meshes):
            scene.add_geometry(mesh, node_name=f"frame_{i:03d}")

        return scene

    def validate_folder_structure(self):
        storage = self.storage_options
        error_message = "Could not access folder {}. Either: it does not exist; " \
                        "there was a typo in the path; or Python does not have sufficient privileges.".format

        assert os.path.isdir(storage.base_path), error_message(storage.base_path)
        assert os.path.isdir(storage.colour_path), error_message(storage.colour_path)
        assert os.path.isdir(storage.depth_path), error_message(storage.depth_path)
        assert os.path.isdir(storage.mask_path) or self.should_create_masks, \
            f"Could not access mask folder {storage.mask_path}. Either: it does not exist; " \
            f"there was a typo in the path; Python does not have sufficient privileges; or the path does not exist " \
            f"AND the flag `--create_masks` was not enabled in the CLI."

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

    parser.add_argument('--create_masks', help='Whether to create masks for dynamic objects',
                        action='store_true')
    parser.add_argument('--fps', type=int, help='The frame rate of the input sequence.', default=60)
    parser.add_argument('--include_background', help='Include the background in the reconstructed mesh.',
                        action='store_true')
    parser.add_argument('--static_background', help='Whether to generate a static background.',
                        action='store_true')
    parser.add_argument('--num_frames', type=int, help='The maximum of frames to process. '
                                                       'Set to -1 (default) to process all frames.', default=-1)
    parser.add_argument('--estimate_depth', action='store_true',
                        help='Flag to indicate that depth maps estimated by a neural network model should be used '
                             'instead of the ground truth depth maps.')
    parser.add_argument('--estimate_camera_params', action='store_true',
                        help='Flag to indicate that camera intrinsic and extrinsic parameters estimated with COLMAP '
                             'should be used instead of the ground truth parameters (if they exist).')

    # TODO: Use the class `UnrealDatasetInfo' to load the dataset info from disk, rather than using CLI args.
    StorageOptions.add_args(parser)
    DepthOptions.add_args(parser)
    MaskDilationOptions.add_args(parser)
    MeshFilteringOptions.add_args(parser)
    MeshDecimationOptions.add_args(parser)
    COLMAPOptions.add_args(parser)

    args = parser.parse_args()
    print(args)

    storage_options = StorageOptions.from_args(args)
    depth_options = DepthOptions.from_args(args)
    filtering_options = MeshFilteringOptions.from_args(args)
    dilation_options = MaskDilationOptions.from_args(args)
    decimation_options = MeshDecimationOptions.from_args(args)
    colmap_options = COLMAPOptions.from_args(args)

    program = Video2Mesh(storage_options,
                         decimation_options=decimation_options, dilation_options=dilation_options,
                         filtering_options=filtering_options, depth_options=depth_options,
                         colmap_options=colmap_options,
                         num_frames=args.num_frames, fps=args.fps,
                         include_background=args.include_background,
                         should_create_masks=args.create_masks,
                         static_background=args.static_background,
                         estimate_depth=args.estimate_depth,
                         estimate_camera_params=args.estimate_camera_params)
    program.run()
