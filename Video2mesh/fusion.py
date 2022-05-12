from typing import Optional, Iterable

import numpy as np
import trimesh
from tqdm import tqdm

from Video2mesh.geometry import dilate_mask, pose_vec2mat
from Video2mesh.io import VTMDataset
from Video2mesh.options import StaticMeshOptions, MaskDilationOptions
from Video2mesh.utils import log
from thirdparty.tsdf_fusion_python import fusion


def tsdf_fusion(dataset: VTMDataset, num_frames=-1, options=StaticMeshOptions(),
                frame_set: Optional[Iterable[int]] = None) -> trimesh.Trimesh:
    """

    :param dataset:
    :param num_frames:
    :param options:
    :param frame_set:
    :return:
    """
    # TODO: Complete `tsdf_fusion(...)` docstring.
    if num_frames == -1:
        num_frames = dataset.num_frames

    log("Estimating voxel volume bounds...")
    vol_bnds = np.zeros((3, 2))

    # Dilate (increase size) of masks so that parts of the dynamic objects are not included in the final mesh
    # (this typically results in floating fragments in the static mesh.)
    mask_dilation_options = MaskDilationOptions(num_iterations=options.depth_mask_dilation_iterations)
    frame_range = frame_set if frame_set else range(num_frames)

    for i in frame_range:
        # Read depth image and camera pose
        mask = dataset.mask_dataset[i]
        mask = dilate_mask(mask, mask_dilation_options)
        depth_im = dataset.depth_dataset[i]
        depth_im[mask > 0] = 0.0
        cam_pose = pose_vec2mat(dataset.camera_trajectory[i])  # 4x4 rigid transformation matrix

        # Compute camera view frustum and extend convex hull
        view_frust_pts = fusion.get_view_frustum(depth_im, dataset.camera_matrix, cam_pose)
        vol_bnds[:, 0] = np.minimum(vol_bnds[:, 0], np.amin(view_frust_pts, axis=1))
        vol_bnds[:, 1] = np.maximum(vol_bnds[:, 1], np.amax(view_frust_pts, axis=1))

    # ======================================================================================================== #
    # Integrate
    # ======================================================================================================== #
    if options.sdf_num_voxels:
        # actual_num_voxels = np.ceil(np.product((vol_bnds[:, 1] - vol_bnds[:, 0]) / options.sdf_voxel_size))
        voxel_size = (np.product(vol_bnds[:, 1] - vol_bnds[:, 0]) / options.sdf_num_voxels) ** (1 / 3)
    else:
        voxel_size = options.sdf_voxel_size

    log("Initializing voxel volume...")
    tsdf_vol = fusion.TSDFVolume(vol_bnds, voxel_size=voxel_size)

    log("Fusing frames...")

    for i in tqdm(frame_range):
        color_image = dataset.rgb_dataset[i]
        mask = dataset.mask_dataset[i]
        mask = dilate_mask(mask, mask_dilation_options)
        depth_im = dataset.depth_dataset[i]
        depth_im[mask > 0] = 0.0
        cam_pose = pose_vec2mat(dataset.camera_trajectory[i])

        # Integrate observation into voxel volume (assume color aligned with depth)
        tsdf_vol.integrate(color_image, depth_im, dataset.camera_matrix, cam_pose, obs_weight=1.)

    # Get mesh from voxel volume and save to disk (can be viewed with Meshlab)
    verts, faces, norms, colors = tsdf_vol.get_mesh()

    # TODO: Cleanup mesh for floating fragments (e.g. via connected components analysis).
    # TODO: Fix this. It seems to mess up the order of the face vertices or something.
    # verts, faces = Video2Mesh.cleanup_with_connected_components(verts, faces, is_object=False, min_components=10)
    mesh = trimesh.Trimesh(vertices=verts, faces=faces, vertex_colors=colors, vertex_normals=norms)

    return mesh
