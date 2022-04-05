import argparse
import enum
import json
import os
import shutil
import traceback

import sys
from contextlib import redirect_stderr, redirect_stdout
from os.path import join as pjoin

import itertools
import numpy as np

from Video2mesh.options import Video2MeshOptions, StorageOptions, DepthOptions, MeshFilteringOptions, \
    MaskDilationOptions, MeshDecimationOptions, COLMAPOptions, StaticMeshOptions
from Video2mesh.utils import log
from Video2mesh.video2mesh import Video2Mesh


def get_default_config() -> dict:
    return {
        'base_path': 'data/rgbd_dataset_freiburg3_walking_xyz',
        'overwrite_ok': True,
        'create_masks': True,
        'num_frames': -1,
        'include_background': False,
        'static_background': False,
        'estimate_depth': False,
        'estimate_camera_params': False,
        'refine_colmap_poses': False,
        # Depth Estimation Options
        'depth_estimation_model': 'adabins',
        'max_depth': 10.0,
        'depth_format': 'depth_to_plane',
        'depth_dtype': 'uint16',
        'sampling_framerate': 6,
        # Mask Dilation Options
        'dilate_mask_iter': 3,
        # Mesh Filtering Options
        'max_depth_dist': 0.1,
        'max_pixel_dist': 2,
        'min_num_components': 5,
        # Mesh Decimation Options
        'num_vertices_background': 16384,
        'num_vertices_object': 1024,
        'decimation_max_error': 0.001,
        # Mesh Reconstruction Options
        'mesh_reconstruction_method': 'tsdf_fusion',
        'depth_mask_dilation_iterations': 32,
        'sdf_volume_size': 10.0,
        'sdf_voxel_size': 0.02,
        # COLMAP Options
        'multiple_cameras': False,
        'dense': False,
        'quality': 'low',
        'use_raw_pose': False,
        'binary_path': '/usr/local/bin/colmap',
        'vocab_path': '/root/.cache/colmap/vocab.bin',
        # Export Options
        'webxr_path': 'thirdparty/webxr3dvideo/docs',
        'webxr_url': 'http://localhost:8080',
    }


class CameraParamsType(enum.Enum):
    GROUND_TRUTH = enum.auto()
    COLMAP = enum.auto()
    COLMAP_BUNDLE_FUSION = enum.auto()

    @classmethod
    def get_all(cls):
        return [cls.GROUND_TRUTH, cls.COLMAP, cls.COLMAP_BUNDLE_FUSION]


class DepthMapSource(enum.Enum):
    GROUND_TRUTH = enum.auto()
    ADABINS = enum.auto()
    CVDE = enum.auto()

    @classmethod
    def get_all(cls):
        return [cls.GROUND_TRUTH, cls.ADABINS, cls.CVDE]


class MeshReconstructionType(enum.Enum):
    DEPTH_STATIC = enum.auto()
    DEPTH = enum.auto()
    TSDF_FUSION = enum.auto()
    BUNDLE_FUSION = enum.auto()

    @classmethod
    def get_all(cls):
        return [cls.DEPTH_STATIC, cls.DEPTH, cls.TSDF_FUSION, cls.BUNDLE_FUSION]


def main():
    log("Parsing command line arguments...")
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_path', type=str, help='The folder to save the results to.',
                        default='data/grid_search_results')
    parser.add_argument('--data_path', type=str, help="The folder that contains the dataset(s).", default='data')
    args = parser.parse_args()

    log("Creating results folder...")
    results_folder = args.output_path

    if os.path.isdir(results_folder):
        raise RuntimeError(f"The results folder {results_folder} already exists. "
                           f"If you intend to overwrite these files, please delete the folder.")

    os.makedirs(results_folder)

    log("Creating master configuration file...")
    datasets = ['rgbd_dataset_freiburg3_walking_xyz', 'rgbd_dataset_freiburg3_sitting_static']
    camera_param_sources = CameraParamsType.get_all()
    depth_map_sources = DepthMapSource.get_all()
    reconstruction_methods = MeshReconstructionType.get_all()

    master_config = {
        'datasets': datasets,
        'camera_param_sources': [source.name for source in camera_param_sources],
        'depth_map_sources': [source.name for source in depth_map_sources],
        'reconstruction_methods': [method.name for method in reconstruction_methods],
    }
    master_config_path = pjoin(results_folder, 'master_config.json')

    with open(master_config_path, 'w') as f:
        json.dump(master_config, f)

    log(f"Master configuration file saved to {master_config_path}")

    log("Creating individual configurations...")
    configs = dict()

    for (dataset, camera_param_source, depth_map_source, reconstruction_method) in \
            itertools.product(datasets, camera_param_sources, depth_map_sources, reconstruction_methods):
        dataset_path = pjoin(args.data_path, dataset)

        if not os.path.isdir(dataset_path):
            raise RuntimeError(f"Could not find the folder {dataset_path}.")

        config = get_default_config()
        config['num_frames'] = 150
        config['base_path'] = dataset_path

        if camera_param_source == CameraParamsType.GROUND_TRUTH:
            config['estimate_camera_params'] = False
        elif camera_param_source == CameraParamsType.COLMAP:
            config['estimate_camera_params'] = True
        elif camera_param_source == CameraParamsType.COLMAP_BUNDLE_FUSION:
            if reconstruction_method != MeshReconstructionType.BUNDLE_FUSION:
                raise RuntimeError(
                    f"Invalid configuration. The camera param source {camera_param_source.name} must be used with the "
                    f"mesh reconstruction method {MeshReconstructionType.BUNDLE_FUSION.name}, "
                    f"but got {reconstruction_method.name} instead."
                )

            config['estimate_camera_params'] = True
            config['refine_colmap_poses'] = True
        else:
            raise RuntimeError(f"Unsupported camera param source {camera_param_source.name}.")

        if depth_map_source == DepthMapSource.GROUND_TRUTH:
            config['estimate_depth'] = False
        elif depth_map_source == DepthMapSource.ADABINS:
            config['estimate_depth'] = True
            config['depth_estimation_model'] = 'adabins'
        elif depth_map_source == DepthMapSource.CVDE:
            config['estimate_depth'] = True
            config['depth_estimation_model'] = 'cvde'
            config['sampling_framerate'] = 6
        else:
            raise RuntimeError(f"Unsupported depth map source {depth_map_source.name}.")

        if reconstruction_method == MeshReconstructionType.DEPTH_STATIC:
            config['include_background'] = True
            config['static_background'] = True
        elif reconstruction_method == MeshReconstructionType.DEPTH:
            config['include_background'] = True
            config['static_background'] = False
        elif reconstruction_method == MeshReconstructionType.TSDF_FUSION:
            config['include_background'] = False
            config['static_background'] = False
            config['mesh_reconstruction_method'] = 'tsdf_fusion'
        elif reconstruction_method == MeshReconstructionType.BUNDLE_FUSION:
            config['include_background'] = False
            config['static_background'] = False
            config['mesh_reconstruction_method'] = 'bundle_fusion'
        else:
            raise RuntimeError(f"Unsupported reconstruction method {reconstruction_method.name}.")

        config_name = f"{dataset}-{camera_param_source.name}-{depth_map_source.name}-{reconstruction_method.name}"
        config_folder = pjoin(results_folder, config_name)
        os.makedirs(config_folder)

        config_path = pjoin(config_folder, 'config.json')

        with open(config_path, 'w') as f:
            json.dump(config, f)

        configs[config_name] = config

        log(f"Created config '{config_name}' at {config_folder}.")

        log(f"Running pipeline with config {config_name}...")
        pipeline_args = argparse.Namespace(**config)

        log_file = pjoin(config_folder, 'log.txt')
        log(f"Check the file {log_file} for detailed output.")

        with open(log_file, 'w') as f, redirect_stdout(f), redirect_stderr(f):
            try:
                video2mesh_options = Video2MeshOptions.from_args(pipeline_args)
                storage_options = StorageOptions.from_args(pipeline_args)
                depth_options = DepthOptions.from_args(pipeline_args)
                filtering_options = MeshFilteringOptions.from_args(pipeline_args)
                dilation_options = MaskDilationOptions.from_args(pipeline_args)
                decimation_options = MeshDecimationOptions.from_args(pipeline_args)
                colmap_options = COLMAPOptions.from_args(pipeline_args)
                static_mesh_options = StaticMeshOptions.from_args(pipeline_args)

                program = Video2Mesh(options=video2mesh_options,
                                     storage_options=storage_options,
                                     decimation_options=decimation_options,
                                     dilation_options=dilation_options,
                                     filtering_options=filtering_options,
                                     depth_options=depth_options,
                                     colmap_options=colmap_options,
                                     static_mesh_options=static_mesh_options)
                program.run()
            except:
                traceback.print_exc()
                log(f"Encountered error, check the logs for more details.", file=sys.stderr)

                continue

        log(f"Copying output to results folder...")

        with open(os.devnull, 'w') as f, redirect_stdout(f):
            dataset = program._get_dataset()

        output_matrix_path = pjoin(config_folder, dataset.camera_matrix_filename)
        output_trajectory_path = pjoin(config_folder, dataset.camera_trajectory_filename)

        if camera_param_source == CameraParamsType.GROUND_TRUTH:
            shutil.copy(pjoin(dataset.base_path, dataset.camera_matrix_filename),
                        output_matrix_path)
            shutil.copy(pjoin(dataset.base_path, dataset.camera_trajectory_filename),
                        output_trajectory_path)
        elif camera_param_source == CameraParamsType.COLMAP:
            shutil.copy(pjoin(dataset.base_path, dataset.estimated_camera_matrix_filename),
                        output_matrix_path)
            shutil.copy(pjoin(dataset.base_path, dataset.estimated_camera_trajectory_filename),
                        output_trajectory_path)
        elif camera_param_source == CameraParamsType.COLMAP_BUNDLE_FUSION:
            shutil.copy(pjoin(dataset.base_path, dataset.estimated_camera_matrix_filename),
                        output_matrix_path)

            trajectory = program._refine_colmap_poses(dataset)
            np.savetxt(output_trajectory_path, trajectory)

        shutil.copytree(pjoin(dataset.base_path, program.mesh_folder), pjoin(config_folder, program.mesh_folder))


if __name__ == '__main__':
    main()
