import datetime
import os
import shutil
import subprocess
from multiprocessing.pool import ThreadPool
from os.path import join as pjoin

import imageio
import numpy as np
import psutil
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from tqdm import tqdm

from Video2mesh.image_processing import dilate_mask
from Video2mesh.io import VTMDataset, ImageFolderDataset
from Video2mesh.fusion import BundleFusionConfig
from Video2mesh.options import StaticMeshOptions, MaskDilationOptions, DepthOptions, DepthEstimationModel
from Video2mesh.utils import log
from thirdparty.consistent_depth.depth_fine_tuning import DepthFineTuner
from thirdparty.consistent_depth.loaders.video_dataset import VideoFrameDataset
from thirdparty.consistent_depth.monodepth.depth_model_registry import get_depth_model
from thirdparty.consistent_depth.params import Video3dParamsParser
from thirdparty.consistent_depth.process import DatasetProcessor
from thirdparty.consistent_depth.utils.torch_helpers import to_device


class CVDEExperimentDataset(VTMDataset):
    def use_estimated_depth(self, depth_options=DepthOptions()) -> 'CVDEExperimentDataset':
        super().use_estimated_depth(depth_options)

        if depth_options.depth_estimation_model == DepthEstimationModel.CVDE:
            self._estimate_depth_cvde(self.path_to_estimated_depth_maps,
                                      sampling_framerate=depth_options.sampling_framerate)

        return self

    def _estimate_depth_cvde(self, path_to_estimated_depth, sampling_framerate=3):
        super()._estimate_depth_cvde(path_to_estimated_depth, sampling_framerate)

        results_path = pjoin(self.base_path, self.cvde_folder)

        video_filename = "video.mp4"
        video_path = pjoin(results_path, video_filename)
        args = [
            '--video_file', video_path,
            '--path', results_path,
            '--camera_model', "PINHOLE",
            '--camera_params', f"{self.fx}, {self.fy}, {self.cx}, {self.cy}",
            '--batch_size', str(2),
            '--num_epochs', str(40),
            '--make_video',
            '--op', 'extract_frames'
        ]
        parser = Video3dParamsParser()
        params = parser.parse(args)

        dp = DatasetProcessor()
        dp.process(params)

        depth_fine_tuner = DepthFineTuner(dp.out_dir, range(self.num_frames), params)
        weights_folder = pjoin(depth_fine_tuner.out_dir, 'checkpoints')

        checkpoints = sorted(os.listdir(weights_folder))
        tmp_folder = pjoin(depth_fine_tuner.out_dir, 'tmp')
        tmp_depth_folder = pjoin(tmp_folder, 'depth')
        tmp_masked_depth_folder = pjoin(tmp_folder, 'masked_depth')
        os.makedirs(tmp_depth_folder, exist_ok=True)
        os.makedirs(tmp_masked_depth_folder, exist_ok=True)

        viz_path = pjoin(depth_fine_tuner.out_dir, 'viz')
        os.makedirs(viz_path, exist_ok=True)

        dataset = VideoFrameDataset(pjoin(self.base_path, self.rgb_folder, '{:06d}.png'), range(self.num_frames))
        data_loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4)
        fps = int(self.metadata.fps)

        for i in [5 * 2 ** p for p in range(6) if 5 * 2 ** p <= len(checkpoints)]:
            depth_map_index = 0

            weights_path = pjoin(weights_folder, checkpoints[i - 1])
            model = get_depth_model(params.model_type)(model_path_override=weights_path)
            model.eval()

            with tqdm(total=len(dataset)) as progress_bar:
                for data in data_loader:
                    data = to_device(data)
                    stacked_images, metadata = data

                    depth_maps = model.forward(stacked_images, metadata)

                    depth_map = depth_maps.detach().cpu().numpy().squeeze()
                    depth_map = depth_map / 10.0

                    output_path = os.path.join(tmp_depth_folder, f'{depth_map_index:06d}.png')
                    plt.imsave(output_path, depth_map, cmap='magma_r', vmin=0.0, vmax=1.0)

                    depth_map_index += 1
                    progress_bar.update(1)

            video_path = pjoin(viz_path, f"{i:02d}_epochs.mp4")
            ffmpeg_cmd = f"ffmpeg -i {pjoin(tmp_depth_folder, '%06d.png')} -r {fps} -vcodec libx264 -pix_fmt yuv420p -y {video_path}"
            log(ffmpeg_cmd)
            #  Have to pass command as single string and use shell=True to avoid issues with quote escaping.
            subprocess.run(ffmpeg_cmd, shell=True).check_returncode()

            options = StaticMeshOptions()
            self.create_masked_depth(MaskDilationOptions(num_iterations=options.depth_mask_dilation_iterations))

            start = datetime.datetime.now()

            log(f"Creating masked depth maps at {tmp_masked_depth_folder}")

            pool = ThreadPool(processes=psutil.cpu_count(logical=False))
            log("Create thread pool")
            dilation_options = MaskDilationOptions(num_iterations=options.depth_mask_dilation_iterations)

            def save_depth(j, depth_map, mask):
                binary_mask = mask > 0.0
                binary_mask = dilate_mask(binary_mask, dilation_options)

                depth_map[binary_mask] = 0.0
                depth_map = depth_map / self.depth_scaling_factor  # undo depth scaling done during loading.
                depth_map = depth_map.astype(np.uint16)
                output_path = pjoin(tmp_masked_depth_folder, f"{j:06d}.png")
                imageio.imwrite(output_path, depth_map)

                log(f"Writing masked depth to {output_path}")

            tmp_depth_dataset = ImageFolderDataset(self.path_to_estimated_depth_maps,
                                                   transform=self._get_depth_map_transform())
            pool.starmap(save_depth, zip(range(len(self)), tmp_depth_dataset, self.mask_dataset))

            elapsed = datetime.datetime.now() - start

            log(f"Created {len(os.listdir(tmp_masked_depth_folder))} masked depth maps in {elapsed}")

            dataset_path = os.path.abspath(self.base_path)

            bundle_fusion_path = os.environ['BUNDLE_FUSION_PATH']
            default_config_path = pjoin(bundle_fusion_path, 'zParametersDefault.txt')
            config = BundleFusionConfig.load(default_config_path)
            config['s_SDFMaxIntegrationDistance'] = options.sdf_volume_size
            config['s_SDFVoxelSize'] = options.sdf_voxel_size
            config['s_cameraIntrinsicFx'] = int(self.fx)
            config['s_cameraIntrinsicFy'] = int(self.fy)
            config['s_cameraIntrinsicCx'] = int(self.cx)
            config['s_cameraIntrinsicCy'] = int(self.cy)

            bundle_fusion_output_dir = pjoin(dataset_path, 'bundle_fusion')
            config['s_generateMeshDir'] = bundle_fusion_output_dir

            config_output_path = pjoin(dataset_path, 'bundleFusionConfig.txt')
            config.save(config_output_path)

            bundle_fusion_bin = os.environ['BUNDLE_FUSION_BIN']
            bundling_config_path = pjoin(bundle_fusion_path, 'zParametersBundlingDefault.txt')
            bundling_config = BundleFusionConfig.load(bundling_config_path)

            submap_size = bundling_config['s_submapSize']
            bundling_config['s_maxNumImages'] = (
                                                            self.num_frames + submap_size) // submap_size  # the `+ submap_size` is to avoid 'off-by-one' like errors.

            bundling_config_output_path = pjoin(dataset_path, 'bundleFusionBundlingConfig.txt')
            bundling_config.save(bundling_config_output_path)

            # BundleFusion prepends the dataset path, so we need to give the depth folder relative to the dataset path.
            depth_folder_for_bundle_fusion = os.path.relpath(tmp_masked_depth_folder, dataset_path)

            return_code = subprocess.call([bundle_fusion_bin, config_output_path, bundling_config_output_path,
                                           dataset_path, depth_folder_for_bundle_fusion])

            if return_code:
                raise RuntimeError(f"BundleFusion returned a non-zero code, check the logs for what went wrong.")

            # Read ply file into trimesh object
            mesh_path = pjoin(bundle_fusion_output_dir, 'mesh.ply')
            shutil.copy(mesh_path, pjoin(viz_path, f"{i:02d}_epochs.ply"))

        shutil.rmtree(tmp_folder)
