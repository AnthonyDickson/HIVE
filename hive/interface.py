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

import gradio as gr
from argparse import ArgumentParser

from hive.options import BackgroundMeshOptions, COLMAPOptions, MaskDilationOptions, MeshDecimationOptions, \
    MeshFilteringOptions, MeshReconstructionMethod, PipelineOptions, StorageOptions, WebXROptions, InpaintingMode
from hive.pipeline import Pipeline


class Interface:
    quality_choices = ('low', 'medium', 'high', 'extreme')

    @staticmethod
    def get_interface() -> gr.Blocks:
        def start_pipeline(o):
            options = PipelineOptions(num_frames=int(o[num_frames]), frame_step=int(o[frame_step]),
                                      estimate_pose=o[estimate_pose], estimate_depth=o[estimate_depth],
                                      background_only=o[background_only], static_camera=o[static_camera],
                                      align_scene=o[align_scene],
                                      inpainting_mode=InpaintingMode.from_integer(int(o[use_inpainting])),
                                      billboard=o[billboard], disable_scaling=o[disable_scaling],
                                      disable_coverage_constraint=o[disable_coverage_constraint],
                                      log_file=o[log_file])
            storage_options = StorageOptions(dataset_path=o[dataset_path], output_path=o[output_path],
                                             overwrite_ok=o[overwrite_ok], no_cache=o[no_cache])
            decimation_options = MeshDecimationOptions(num_faces_background=int(o[num_faces_background]),
                                                       num_faces_object=int(o[num_faces_object]),
                                                       max_error=o[decimation_max_error])
            dilation_options = MaskDilationOptions(num_iterations=int(o[dilate_mask_iter]))
            filtering_options = MeshFilteringOptions(max_pixel_distance=int(o[max_pixel_dist]),
                                                     max_depth_distance=o[max_depth_dist],
                                                     min_num_components=int(o[min_num_components]))
            colmap_options = COLMAPOptions(is_single_camera=o[is_single_camera], dense=o[dense], quality=o[quality],
                                           binary_path=o[binary_path], vocab_path=o[vocab_path])
            static_mesh_options = BackgroundMeshOptions(
                reconstruction_method=MeshReconstructionMethod.from_string(o[mesh_reconstruction_method]),
                depth_mask_dilation_iterations=int(o[depth_mask_dilation_iterations]),
                sdf_volume_size=float(o[sdf_volume_size]), sdf_voxel_size=float(o[sdf_voxel_size]),
                sdf_max_voxels=int(o[sdf_max_voxels]))
            webxr_options = WebXROptions(webxr_source_path=o[webxr_source_path], webxr_path=o[webxr_path],
                                         webxr_url=o[webxr_url],
                                         webxr_add_ground_plane=o[webxr_add_ground_plane],
                                         webxr_add_sky_box=o[webxr_add_sky_box],
                                         webxr_run_server=o[webxr_run_server])

            print("Running the pipeline with the following configuration: ", options, storage_options,
                  decimation_options, dilation_options, filtering_options, colmap_options, static_mesh_options,
                  webxr_options)

            pipeline = Pipeline(options=options, storage_options=storage_options, decimation_options=decimation_options,
                                dilation_options=dilation_options, filtering_options=filtering_options,
                                colmap_options=colmap_options, static_mesh_options=static_mesh_options,
                                webxr_options=webxr_options)
            pipeline.run()

        with gr.Blocks(theme=gr.themes.Default()) as demo:
            with gr.Accordion("Quickstart", open=True):
                with gr.Row():
                    gr.Markdown(
                        """For a more detailed explanation, refer to the [README](https://github.com/AnthonyDickson/HIVE/blob/master/README.md).
                        
                        # Quickstart
                        1. Fill in the CLI options below.
                        2. Click the button at the bottom of the page that says 'Start Pipeline'.
                        3. When the pipeline is finished running, it will give you a link (check the terminal output). 
                           Click on this link to view the 3D video.
                        
                        # Common CLI Options
                        - `--dataset_path <path/to/dataset>` Specify the path to either: a video file, TUM dataset or an iPhone dataset (StrayScanner).
                        - `--output_path <path/to/folder>` Specify where the results should be written to.
                        - `--overwrite_ok` Allow existing video files in `output_path` or the WebXR export path to be overwritten.
                        - `--no_cache` By default the pipeline will use any cached converted datasets in `output_path`. Use this flag to automatically delete any cached datasets.
                        - `--estimate_depth` By default the pipeline will try to use any depth maps in the `depth` folder. Use this flag to use estimated depth maps instead.
                        - `--estimate_pose` By default the pipeline will try to use ground truth camera intrinsics matrix and poses in the `camera_matrix.txt` and `camera_trajectory.txt` files. Use this flag to use COLMAP to estimate the camera parameters instead.
                        - `--num_frames <int>` If specified, any frames after this index are truncated.
                        - `--align_scene` Whether to align the scene with the ground plane. Enable this if the recording device was held at an angle (facing upwards or downwards, not level) and the scene is not level in the renderer. This setting is recommended if you are using estimated pose.
                        - `--inpainting_mode` Use Lama to inpaint the background.
                            - `0` - no inpainting.
                            - `1` - Depth: cv2, Background: cv2
                            - `2` - Depth: cv2, Background: LaMa
                            - `3` - Depth: LaMa, Background: cv2
                            - `4` - Depth: LaMa, Background: LaMa
                        - `--billboard` Creates flat billboards for foreground objects. This is intended as a workaround for cases where the estimated depth results in stretched out meshes with missing body parts.
                        - `--static_camera` Indicate that the camera was not moving during capture. This will use the Kinect sensor camera matrix and the identity pose for the camera trajectory. Note: You do not need the flag `--estimate_pose` when using this flag.
                        - `--webxr_run_server` Whether to automatically run the renderer web server.
                        """
                    )

            with gr.Accordion("Storage Options", open=True):
                with gr.Row():
                    with gr.Column():
                        dataset_path = gr.Text(value="/app/data/your_dataset_here", label="dataset_path",
                                               interactive=True)
                    with gr.Column():
                        output_path = gr.Text(value="/app/outputs/output_folder_name_here", label="output_path",
                                              interactive=True)
                    with gr.Column():
                        overwrite_ok = gr.Checkbox(value=False, label="overwrite_ok", interactive=True)
                        no_cache = gr.Checkbox(value=False, label="no_cache", interactive=True)

            with gr.Accordion("Pipeline Options", open=True):
                with gr.Row():
                    with gr.Column():
                        num_frames = gr.Number(value=-1, label="num_frames", interactive=True, minimum=-1, precision=0)
                        frame_step = gr.Number(value=15, label="frame_step", interactive=True, minimum=1, precision=0)
                    with gr.Column():
                        use_inpainting = gr.Dropdown(
                            choices=[mode.name for mode in InpaintingMode.get_modes()],
                            value=InpaintingMode.get_name(0), type="index",
                            multiselect=False, label="inpainting_mode", interactive=True
                        )
                        log_file = gr.Text(value="logs.log", label="log_file", interactive=True)

                    with gr.Column():
                        estimate_pose = gr.Checkbox(value=True, label="estimate_pose", interactive=True)
                        estimate_depth = gr.Checkbox(value=True, label="estimate_depth", interactive=True)
                        background_only = gr.Checkbox(value=False, label="background_only", interactive=True)
                        align_scene = gr.Checkbox(value=False, label="align_scene", interactive=True)
                        billboard = gr.Checkbox(value=False, label="billboard", interactive=True)
                        static_camera = gr.Checkbox(value=False, label="static_camera", interactive=True)
                        disable_scaling = gr.Checkbox(value=False, label="disable_scaling", interactive=True)
                        disable_coverage_constraint = gr.Checkbox(value=False, label="disable_coverage_constraint",
                                                                  interactive=True)

            with gr.Accordion("WebXROptions", open=False):
                with gr.Row():
                    with gr.Column():
                        webxr_source_path = gr.Text(value='third_party/HIVE_Renderer', label="webxr_source_path",
                                             interactive=True)
                        webxr_path = gr.Text(value='third_party/HIVE_Renderer/docs/video/', label="webxr_path",
                                             interactive=True)
                    with gr.Column():
                        webxr_url = gr.Text(value='http://localhost:8080', label="webxr_url", interactive=True)
                        webxr_run_server = gr.Checkbox(value=True, label="webxr_run_server", interactive=True)
                    with gr.Column():
                        webxr_add_ground_plane = gr.Checkbox(value=False, label="webxr_add_ground_plane",
                                                             interactive=True)
                        webxr_add_sky_box = gr.Checkbox(value=False, label="webxr_add_sky_box", interactive=True)

            with gr.Accordion("Static Mesh Options", open=False):
                with gr.Row():
                    with gr.Column():
                        mesh_reconstruction_method = gr.Dropdown(choices=[method.get_cli_name() for method in
                                                                          BackgroundMeshOptions.supported_reconstruction_methods],
                                                                 value='tsdf_fusion',
                                                                 label="mesh_reconstruction_method", interactive=True)
                    with gr.Column():
                        depth_mask_dilation_iterations = gr.Number(value=32, label="depth_mask_dilation_iterations",
                                                                   interactive=True, minimum=0, precision=0)
                    with gr.Column():
                        sdf_volume_size = gr.Number(value=5.0, label="sdf_volume_size", interactive=True, minimum=1e-8)
                    with gr.Column():
                        sdf_voxel_size = gr.Number(value=0.02, label="sdf_voxel_size", interactive=True, minimum=1e-8)
                    with gr.Column():
                        sdf_max_voxels = gr.Number(value=80_000_000, label="sdf_max_voxels", interactive=True,
                                                   minimum=1, precision=0)

            with gr.Accordion("Mesh Filtering Options", open=False):
                with gr.Row():
                    with gr.Column():
                        max_depth_dist = gr.Number(value=0.1, label="max_depth_dist", interactive=True, minimum=0.0)
                    with gr.Column():
                        max_pixel_dist = gr.Number(value=2.0, label="max_pixel_dist", interactive=True, minimum=0.0)
                    with gr.Column():
                        min_num_components = gr.Number(value=5, label="min_num_components", interactive=True,
                                                       minimum=1, precision=0)

            with gr.Accordion("Mask Dilation Options", open=False):
                with gr.Row():
                    with gr.Column():
                        dilate_mask_iter = gr.Number(value=0, label="dilate_mask_iter", interactive=True,
                                                     minimum=0, precision=0)

            with gr.Accordion("Mesh Decimation Options", open=False):
                with gr.Row():
                    with gr.Column():
                        num_faces_background = gr.Number(value=2 ** 14, label="num_faces_background",
                                                         interactive=True, minimum=1, precision=0)
                    with gr.Column():
                        num_faces_object = gr.Number(value=2 ** 10, label="num_faces_object", interactive=True,
                                                     minimum=1, precision=0)
                    with gr.Column():
                        decimation_max_error = gr.Number(value=0.001, label="decimation_max_error", interactive=True,
                                                         minimum=0.)

            with gr.Accordion("COLMAP Options", open=False):
                with gr.Row():
                    with gr.Column():
                        is_single_camera = gr.Checkbox(value=True, label="is_single_camera", interactive=True)
                        dense = gr.Checkbox(value=False, label="dense", interactive=True)
                    with gr.Column():
                        quality = gr.Dropdown(choices=Interface.quality_choices, value="low", multiselect=False,
                                              label="quality", interactive=True)
                    with gr.Column():
                        binary_path = gr.Text(value="/usr/local/bin/colmap", label="binary_path", interactive=True)
                    with gr.Column():
                        vocab_path = gr.Text(value="/root/.cache/colmap/vocab.bin", label="vocab_path",
                                             interactive=True)

            btn = gr.Button(value="Start Pipeline")
            inputs = {dataset_path, output_path, overwrite_ok, no_cache, num_frames, frame_step, use_inpainting,
                      log_file, estimate_pose, estimate_depth, background_only, align_scene, billboard, static_camera,
                      disable_scaling, disable_coverage_constraint, webxr_source_path, webxr_path, webxr_url,
                      webxr_run_server, webxr_add_ground_plane, webxr_add_sky_box, mesh_reconstruction_method,
                      depth_mask_dilation_iterations, sdf_volume_size, sdf_voxel_size, sdf_max_voxels, max_depth_dist,
                      max_pixel_dist, min_num_components, dilate_mask_iter, num_faces_background, num_faces_object,
                      decimation_max_error, is_single_camera, dense, quality, binary_path, vocab_path}
            btn.click(fn=start_pipeline, inputs=inputs, outputs=None)

        demo.title = "HIVE (hive)"

        return demo


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--port', type=int, help='The port number to run the interface web server on.', default=8081)
    args = parser.parse_args()

    port_number = args.port
    print(f"Navigate to http://localhost:{port_number} in your browser to start.")

    interface = Interface.get_interface()
    interface.launch(server_name="0.0.0.0", server_port=port_number)
