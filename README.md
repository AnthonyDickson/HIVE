# HIVE (Home Immersive Video Experience)
HIVE is a program for creating 3D free-viewpoint video from RGB video.

![demo of 3D video](images/demo.gif)

A live demo can be viewed at [anthonydickson.github.io/HIVE_Renderer](https://anthonydickson.github.io/HIVE_Renderer).
# Getting Started
The following instructions assume you are using Docker to run HIVE.

We recommend you to use the standalone Docker image `anthonydickson/hive:standalone` if you are only interested in running HIVE.
It contains everything needed to run HIVE.
If you want to run edited code, refer to [Setting Up Your Development Environment](#setting-up-your-development-environment).
## System Requirements
- Windows or Ubuntu
- NVIDIA GPU with 6 GB+ memory, e.g. RTX 2080 Ti, RTX 3060.
- CUDA 11.6+
- 16 GB of RAM

## Quickstart
Here are the steps to quickly run a video through HIVE. You only need to download (pull) the Docker image.
1. Make sure you have Docker installed and GPU support is enabled. You can check this by running a container with the options `--gpus all` and verifying that the `nvidia-smi` command shows the correct information.
2. Create one folder containing two folders, a folder for your videos/datasets and a folder for the outputs:
    ```shell
    mkdir HIVE
    cd HIVE
    mkdir data
    mkdir outputs
    ```
3. Copy your videos or RGB-D datasets (TUM, StrayScanner or Unreal) into your data folder, e.g. `HIVE/data`.
4. Run the following command in a terminal from inside the root folder, e.g. `HIVE/`:
    ```shell
    GUI_PORT_NUMBER=8081;SERVER_PORT_NUMBER=8080; docker run --name HIVE --rm --gpus all -p ${GUI_PORT_NUMBER}:${GUI_PORT_NUMBER} -p ${SERVER_PORT_NUMBER}:{SERVER_PORT_NUMBER} -v $(pwd)/data:/app/data -v $(pwd)/outputs:/app/outputs -it anthonydickson/hive:standalone python3 -m hive.interface --port ${GUI_PORT_NUMBER}
    ```
5. Navigate to [localhost:8081](http://localhost:8081) and fill in `dataset_path` and `output_path` with the relative path to the dataset in your data folder and the relative path of the output folder in your outputs folder, and click the button at the bottom of the page that says 'Start Pipeline'. 
   You can leave the other settings at their default values.
   For long videos, you can set `num_frames` to something like 150 to make it run faster.
   For datasets with ground truth data, uncheck `estimate_pose` and/or `estimate_depth` to use the ground truth data.
6. After the pipeline has finished running, check the terminal output for a link. Navigate to that link in your web browser to view the 3D video.
7. To run another video, exit the Docker container in your terminal (Ctrl+C).

**Note:** Oculus headset users can use desktop mode in Oculus Link to view the 3D video in VR.
Run the steps above as normal, and at the end click the button at the bottom of the screen saying 'Enter VR' from your headset.

## Cloning the Project
Clone the repo:
```shell
git clone --recurse-submodules https://github.com/AnthonyDickson/HIVE.git 
```
If you forget to or cannot clone with `--recurse-submodules`, then clone the git dependencies with the following:
```shell
git submodule update --init --recursive
```
This command can also be used to pull/update any changes in the submodules. 

## Setting Up Your Development Environment
Choose one of three options for setting up the dev environment (in the recommended order):
1. [Pre-built Docker image](#pre-built-docker-image)
2. [Building the Docker image locally](#building-the-docker-image-locally)
3. [Local installation](#local-installation)

### Pre-Built Docker Image
1. Pull (download) the pre-built image (~18 GB): 
      ```shell
      docker pull anthonydickson/hive:runtime-cu118
      ```
2. Done! Go to [Running the Program](#running-the-program) for basic usage.

### Building the Docker image locally
1. Run the build command:
      ```shell
      docker build -f Dockerfile.runtime -t anthonydickson/hive:runtime-cu118 .
      ```
2. Done! Go to [Running the Program](#running-the-program) for basic usage.

### Local Installation
1. Install Python 3.8. A virtual environment (e.g., Conda, virtualenv, pipenv) is recommended.

2. Install the Python packages:
    ```shell
    pip install -r requirements.txt
    ```
3. Install the dependencies:
   - CUDA Toolkit 11.6+
   - OpenCV 3.4.16 
     - Make sure to enable the CMake flag `-DWITH_CUDA=true`.
   - COLMAP
   - The `gltf_2.0_draco_extension` branch of [Draco](https://github.com/google/draco.git).
   - Download the model weights:
     ```shell
     WEIGHTS_PATH=/root/.cache/pretrained
     COLMAP_VOCAB_PATH=/root/.cache/colmap
     
     mkdir ${WEIGHTS_PATH}
     mkdir ${COLMAP_VOCAB_PATH}
     
     python scripts/download_weights.py
     wget https://github.com/intel-isl/DPT/releases/download/1_0/dpt_hybrid_nyu-2ce69ec7.pt -O ${WEIGHTS_PATH}/dpt_hybrid_nyu.pt
     curl -L $(yadisk-direct https://disk.yandex.ru/d/ouP6l8VJ0HpMZg) -o big-lama.zip && unzip big-lama.zip -d ${WEIGHTS_PATH} && rm big-lama-zip
     wget https://demuc.de/colmap/vocab_tree_flickr100K_words256K.bin -O ${COLMAP_VOCAB_PATH}/vocab.bin
     ```
  
   Refer to the [Dockerfile](Dockerfile.runtime) for detailed setup instructions on Ubuntu 20.04.

4. Done! Go to [Running the Program](#running-the-program) for basic usage.

## Running the Program
### Sample Dataset
You can download a sample dataset from the [TUM website](https://vision.in.tum.de/data/datasets/rgbd-dataset/download).
The sequence `fr3/walking_xyz` is a good one to start with.
Make sure you download and extract the dataset to the `data/` folder.

### Example Usage
The following examples assume you are using the standalone Docker image.

Below is an example of how to run the program with ground truth data:
```shell
docker run --rm --gpus all -v $(pwd)/data:/app/data -v $(pwd)/outputs:/app:outputs -p 8080:8080 -it anthonydickson/hive:standalone python3 -m hive --dataset_path data/rgbd_dataset_freiburg3_walking_xyz --output_path data/rgbd_dataset_freiburg3_walking_xyz_output --num_frames 150 --webxr_run_server
```

Below is an example of how to run the program with estimated data:
```shell
docker run --rm --gpus all -v $(pwd)/data:/app/data -v $(pwd)/outputs:/app:outputs -p 8080:8080 -it anthonydickson/hive:standalone python3 -m hive --dataset_path data/rgbd_dataset_freiburg3_walking_xyz --output_path data/rgbd_dataset_freiburg3_walking_xyz_output --num_frames 150 --frame_step 15 --estimate_pose --estimate_depth --webxr_run_server
```
### Running Examples in a Development Environment
If you are using the cloned repo and either the runtime or dev Docker image, you will need change the commands in this README file by:
- mounting the entire repository folder `-v $(pwd):/app` instead of just the data and outputs folders,
- removing the port bindings (e.g., `-p 8080:8080`),
- removing the flag `--webxr_run_server`,
- running the renderer web server manually (see [Viewing the 3D Video](#viewing-the-3d-video)).

### PyCharm Users
There should be run configurations for PyCharm included when you clone the repo from GitHub in the `.idea` folder.

### CLI Options
If you want help with the CLI and the options, you can either refer to [options.py](hive/options.py) or view the help via:
```shell
docker run --rm -it anthonydickson/hive python3 -m hive --help
```

Below is a list of the most useful CLI options:
- `--dataset_path <path/to/dataset>` Specify the path to either: a video file, TUM dataset or an iPhone dataset (StrayScanner).
- `--output_path <path/to/folder>` Specify where the results should be written to.
- `--overwrite_ok` Allow existing video files in `output_path` or the WebXR export path to be overwritten.
- `--no_cache` By default the pipeline will use any cached converted datasets in `output_path`. Use this flag to automatically delete any cached datasets.
- `--estimate_depth` By default the pipeline will try to use any depth maps that are provided with the input sequence. Use this flag to estimate depth maps instead.
- `--estimate_pose` By default the pipeline will try to use ground truth camera intrinsics matrix and poses in the `camera_matrix.txt` and `camera_trajectory.txt` files. Use this flag to estimate the camera parameters via COLMAP instead.
- `--num_frames <int>` If specified, any frames after this index are truncated.
- `--align_scene` Whether to align the scene with the ground plane. Enable this if the recording device was held at an angle (facing upwards or downwards, not level) and the scene is not level in the renderer. This setting is recommended if you are using estimated pose.
- `--inpainting_mode` Inpaint gaps in the background.
    - `0` - no inpainting.
    - `1` - Depth: cv2, Background: cv2
    - `2` - Depth: cv2, Background: LaMa
    - `3` - Depth: LaMa, Background: cv2
    - `4` - Depth: LaMa, Background: LaMa
- `--billboard` Creates flat billboards for foreground objects. This is intended as a workaround for cases where the estimated depth results in stretched out meshes with missing body parts.
- `--static_camera` Indicate that the camera was not moving during capture. This will use the Kinect sensor camera matrix and the identity pose for the camera trajectory. Note: You do not need the flag `--estimate_pose` when using this flag.
- `--webxr_run_server` Whether to automatically run the renderer web server.

### Docker
The Docker containers will, by default, bring up the python interpreter. All you need to do to get the main script (or any other script) running is to append the usual command, minus the call to python, to the following:
```shell
docker run --rm --gpus all -it anthonydickson/hive:standalone 
```
For example, if you wanted to test whether the container is CUDA enabled: 
```shell
docker run --rm --gpus all -it anthonydickson/hive:standalone python3 -c "import torch; print(torch.cuda.is_available())"
```

### Gradio Web Interface
You can run the pipeline from a web based interface instead of the CLI.
Assuming you are using Docker, you can run this by running the following command:
```shell
docker run -v $(pwd)/data:/app/data -v $(pwd)/outputs:/app/outputs -p 8081:8081 -p 8080:8080 --rm --gpus all -it anthonydickson/hive:standalone python3 -m hive.interface
```
and navigating to [localhost:8081](http://localhost:8081).
Note that you should use relative paths for the dataset path and output paths.

Thank you to Felix for implementing this web interface and the image inpainting.

### Viewing the 3D Video
- Start the Docker container:
  1. If you are using the standalone Docker image, simply add the flag `--webxr_run_server` when you run HIVE.
     If you want to view a video from an output folder, run the following command:
     ```shell
     docker run --rm -p 8080:8080 -v $(pwd)/outputs/<dataset_name>/mesh:/app/third_party/HIVE_Renderer/docs/demo -it anthonydickson/hive:standalone bash -c "cd /app/third_party/HIVE_Renderer && npm run start" 
     ```
     and navigate to [localhost:8080](http://localhost:8080).
  2. If you cloned the repo and are using the Docker image `anthonydickson/hive:runtime-cu118`:
  ```shell
  docker run --rm  --name Hive-Renderer -p 8080:8080 -v $(pwd)/third_party/webxr3dvideo/src:/app/src:ro -v $(pwd)/third_party/webxr3dvideo/docs:/app/docs anthonydickson/hive-renderer:node-16 
  ```
  If you are using PyCharm there is a run configuration included.
- When you run the pipeline it will print the link to view the video, navigate to this link in your browser.

Refer to the [Hive Renderer repo](https://github.com/AnthonyDickson/webxr3dvideo) for the code.

# Data Format
## Input Data Format
This program accepts datasets in three formats:
- TUM [RGB-D SLAM Dataset](https://vision.in.tum.de/data/datasets/rgbd-dataset/file_formats)
- RGB-D datasets created on an iOS device using [StrayScanner](https://apps.apple.com/nz/app/stray-scanner/id1557051662)
- RGB Video
- The HIVE dataset format (see [HIVE Dataset Format](#hive-dataset-format))

The above datasets are automatically converted to the HIVE dataset format.

## Output Format
Each 3D video is saved to a folder with the glTF formatted mesh files and JSON metadata:
```text
<converted dataset>
│   ...
└── mesh
    │   fg.glb
    │   bg.glb
    └── metadata.json
```
This folder is saved under the dataset folder.


## HIVE Dataset Format
Overall, the expected folder structure for the HIVE dataset format is as follows:

```
<dataset>
│   metadata.json
│   camera_matrix.txt
│   camera_trajectory.txt
│   rgb
│   │   000000.png
│   │   000001.png
│   │   ...
│   └── 999999.png
│   depth
│   │   000000.png
│   │   000001.png
│   │   ...
│   └── 999999.png
│   mask
│   │   000000.png
│   │   000001.png
│   │   ...
│   └── 999999.png
└── ...
```

Datasets should be placed in a folder inside the `data/` folder.
Generally, the number of colour frames must match the number of depth maps, masks and lines in the camera trajectory file.
Within each dataset folder, there should be the following 5 items:
1. The metadata in a JSON formatted file that contains the following fields:
   - `num_frames`: The number of frames in the video sequence.
   - `frame_step`: The frequency in frames to sample frames for COLMAP and pose optimisation.
   - `fps`: The framerate of the video.
   - `width`: The width of the video frames in pixels.
   - `height`: The height of the video frames in pixels.
   - `depth_scale`: A scalar that when multiplied with depth map, will transform the depth values to meters.
   - `max_depth`: Depth values are clipped to this value. Roughly corresponds to meters.
   - `depth_mask_dilation_iterations`: The number of times to apply the dilation filter to the dynamic object masks when creating the masked depth maps.
   - `is_gt`: Whether the dataset was created using ground truth camera and depth data.
   - `estimate_pose`: Whether the camera parameters were estimated with COLMAP.
   - `estimate_depth`: Whether the depth maps were estimated.
   - `colmap_options`: The COLMAP configuration that was used if `estimate_pose` was `True`. This is a nested dictionary that contains the following fields:
     - `binary_path`: The path to the COLMAP binary.
     - `vocab_path`: The path to the COLMAP vocab file.
     - `is_single_camera`: Whether the dataset was captured with a single camera.
     - `dense`: Whether to run the dense reconstruction.
     - `quality`: The preset to use for the automatic reconstruction pipeline.
2. The camera intrinsics in a text file in the following format:
   ```text
   fx  0 cx
    0 fy cy
    0  0  1
   ```
   where `fx`, `fy` are the focal length (pixels) and `cx`, `cy` the principal point.
3. The camera trajectory in a text file in the following format:
   ```text
   qx qy qz qw tx ty tz
   ...
   qx qy qz qw tx ty tz
   ```
   where: `qx`, `qy`, `qz` and `qw` form a quaternion; and `tx`, `ty` and `tz` form a translation vector.
   There must be one line per frame in the input video sequence.
   Absolute pose values are expected (i.e. all relative to world origin) that transform points in world coordinates to camera space.
   
   Pose data is stored in the same coordinate frame as COLMAP: x (right), -y (down), -z (forwards, camera looking down +z). 
   To convert to a right-handed coordinate system, you will need to multiply by following homogeneous transform:
    ```
    1  0  0  0
    0 -1  0  0
    0  0 -1  0
    0  0  0  1
    ```
4. The colour (RGB) frames are PNG files in a folder with names that preserve the frames' natural ordering, e.g.:
   ```text
   rgb
   │   000001.png
   │   000002.png
   │   ...
   └── 999999.png
   ```
5. The depth maps are 16-bit PNG files in a folder with names that preserve the frames' natural ordering, e.g.:
   ```text
   depth
   │   000001.png
   │   000002.png
   │   ...
   └── 999999.png
   ```
   The depth maps are expected to be stored in a 16-bit grayscale image. The depth values should be in millimeters and increasing from the camera (i.e. depth = 0 at the camera, depth of 1000 is 1000 millimeters).


# Algorithm Overview
## Dataset Creation
![dataset creation overview](images/vtm_dataset_creation_overview.jpg)

## Mesh Reconstruction
![dynamic mesh reconstruction overview](images/vtm_dynamic_mesh_reconstruction.jpg)
![static mesh reconstruction overview](images/vtm_static_mesh_reconstruction.jpg)

## WebXR Renderer
![webxr renderer overview](images/vtm_renderer_overview.jpg)

# Citing
If you use our code, please cite:
```
@inproceedings{dickson2022vrvideos,
    title={VRVideos: A Flexible Pipeline for VR Video Creation},
    author={Dickson, Anthony and Shanks, Jeremy and Ventura, Jonathan and Knott, Alistair and Zollmann, Stefanie},
    booktitle={2022 IEEE International Conference on Artificial Intelligence and Virtual Reality (AIVR)},
    pages={1--5},
    year={2022},
    organization={IEEE}
}
```
