import logging

import gradio as gr

from video2mesh.options import BackgroundMeshOptions, COLMAPOptions, MaskDilationOptions, MeshDecimationOptions, \
    MeshFilteringOptions, MeshReconstructionMethod, PipelineOptions, StorageOptions, WebXROptions, InpaintingMode
from video2mesh.pipeline import Pipeline


class Interface:
    quality_choices = ('low', 'medium', 'high', 'extreme')

    @staticmethod
    def get_interface() -> gr.Blocks:
        def start_pipeline(o):
            options = PipelineOptions(num_frames=int(o[num_frames]), frame_step=int(o[frame_step]),
                                      estimate_pose=o[estimate_pose], estimate_depth=o[estimate_depth],
                                      background_only=o[background_only], align_scene=o[align_scene],
                                      inpainting_mode=InpaintingMode.from_integer(int(o[use_inpainting])),
                                      use_billboard=o[use_billboard],
                                      log_file=o[log_file])
            storage_options = StorageOptions(dataset_path=o[dataset_path], output_path=o[output_path],
                                             overwrite_ok=o[overwrite_ok], no_cache=o[no_cache])
            decimation_options = MeshDecimationOptions(num_vertices_background=int(o[num_vertices_background]),
                                                       num_vertices_object=int(o[num_vertices_object]),
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
            webxr_options = WebXROptions(webxr_path=o[webxr_path], webxr_url=o[webxr_url],
                                         webxr_add_ground_plane=o[webxr_add_ground_plane],
                                         webxr_add_sky_box=o[webxr_add_sky_box])

            logging.debug("Running the pipeline with the following configuration: ", options, storage_options, decimation_options, dilation_options, filtering_options, colmap_options, static_mesh_options, webxr_options)

            pipeline = Pipeline(options=options, storage_options=storage_options, decimation_options=decimation_options,
                                dilation_options=dilation_options, filtering_options=filtering_options,
                                colmap_options=colmap_options, static_mesh_options=static_mesh_options,
                                webxr_options=webxr_options)
            pipeline.run()

        with gr.Blocks(theme=gr.themes.Default()) as demo:
            with gr.Accordion("Storage Options", open=True):
                with gr.Row():
                    with gr.Column():
                        dataset_path = gr.Text(value="", label="dataset_path", interactive=True)
                    with gr.Column():
                        output_path = gr.Text(value="", label="output_path", interactive=True)
                    with gr.Column():
                        overwrite_ok = gr.Checkbox(value=False, label="overwrite_ok", interactive=True)
                        no_cache = gr.Checkbox(value=False, label="no_cache", interactive=True)

            with gr.Accordion("Pipeline Options", open=True):
                with gr.Row():
                    with gr.Column():
                        num_frames = gr.Number(value=-1, label="num_frames", interactive=True)
                        frame_step = gr.Number(value=15, label="frame_step", interactive=True)
                    with gr.Column():
                        use_inpainting = gr.Dropdown(
                            choices=[mode.name for mode in InpaintingMode.get_modes()],
                            value=InpaintingMode.get_name(0), type="index",
                            multiselect=False, label="inpainting_mode", interactive=True
                        )
                        log_file = gr.Text(value="logs.log", label="log_file", interactive=True)

                    with gr.Column():
                        estimate_pose = gr.Checkbox(value=False, label="estimate_pose", interactive=True)
                        estimate_depth = gr.Checkbox(value=False, label="estimate_depth", interactive=True)
                        background_only = gr.Checkbox(value=False, label="background_only", interactive=True)
                        align_scene = gr.Checkbox(value=False, label="align_scene", interactive=True)
                        use_billboard = gr.Checkbox(value=False, label="use_billboard", interactive=True)

            with gr.Accordion("WebXROptions", open=False):
                with gr.Row():
                    with gr.Column():
                        webxr_path = gr.Text(value='thirdparty/webxr3dvideo/docs', label="webxr_path", interactive=True)
                    with gr.Column():
                        webxr_url = gr.Text(value='http://localhost:8080', label="webxr_url", interactive=True)
                    with gr.Column():
                        webxr_add_ground_plane = gr.Checkbox(value=False, label="webxr_add_ground_plane",
                                                             interactive=True)
                        webxr_add_sky_box = gr.Checkbox(value=False, label="webxr_add_sky_box", interactive=True)

            with gr.Accordion("Static Mesh Options", open=False):
                with gr.Row():
                    with gr.Column():
                        mesh_reconstruction_method = gr.Dropdown(choices=[method.name.lower() for method in
                                                                          BackgroundMeshOptions.supported_reconstruction_methods],
                                                                 value='tsdf_fusion',
                                                                 label="mesh_reconstruction_method", interactive=True)
                    with gr.Column():
                        depth_mask_dilation_iterations = gr.Number(value=32, label="depth_mask_dilation_iterations",
                                                                   interactive=True)
                    with gr.Column():
                        sdf_volume_size = gr.Number(value=5.0, label="sdf_volume_size", interactive=True)
                    with gr.Column():
                        sdf_voxel_size = gr.Number(value=0.02, label="sdf_voxel_size", interactive=True)
                    with gr.Column():
                        sdf_max_voxels = gr.Number(value=80_000_000, label="sdf_max_voxels", interactive=True)

            with gr.Accordion("Mesh Filtering Options", open=False):
                with gr.Row():
                    with gr.Column():
                        max_depth_dist = gr.Number(value=0.1, label="max_depth_dist", interactive=True)
                    with gr.Column():
                        max_pixel_dist = gr.Number(value=2, label="max_pixel_dist", interactive=True)
                    with gr.Column():
                        min_num_components = gr.Number(value=5.0, label="min_num_components", interactive=True)

            with gr.Accordion("Mask Dilation Options", open=False):
                with gr.Row():
                    with gr.Column():
                        dilate_mask_iter = gr.Number(value=0, label="dilate_mask_iter", interactive=True)

            with gr.Accordion("Mesh Decimation Options", open=False):
                with gr.Row():
                    with gr.Column():
                        num_vertices_background = gr.Number(value=2 ** 14, label="num_vertices_background",
                                                            interactive=True)
                    with gr.Column():
                        num_vertices_object = gr.Number(value=2 ** 10, label="num_vertices_object", interactive=True)
                    with gr.Column():
                        decimation_max_error = gr.Number(value=0.001, label="decimation_max_error", interactive=True)

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
            inputs = {num_frames, frame_step, estimate_pose, estimate_depth, background_only, align_scene, log_file,
                      use_inpainting, use_billboard,
                      dataset_path, output_path, overwrite_ok, no_cache,
                      num_vertices_background, num_vertices_object, decimation_max_error,
                      dilate_mask_iter,
                      max_pixel_dist, max_depth_dist, min_num_components,
                      is_single_camera, dense, quality, binary_path, vocab_path,
                      mesh_reconstruction_method, depth_mask_dilation_iterations, sdf_volume_size, sdf_voxel_size,
                      sdf_max_voxels,
                      webxr_path, webxr_url, webxr_add_ground_plane, webxr_add_sky_box}
            btn.click(fn=start_pipeline, inputs=inputs, outputs=None)

        return demo

if __name__ == '__main__':
    interface = Interface.get_interface()
    interface.launch(server_name="0.0.0.0", server_port=8081)
