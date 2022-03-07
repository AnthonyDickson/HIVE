# Video2Mesh
This project looks at creating a 3D video from a RGBD video.
![demo of 3D video](video_3d_demo.gif)
# Getting Started
## Cloning the project
Clone the repo:
```shell
git clone --recurse-submodules https://github.com/eight0153/video2mesh.git 
```
If forget to or cannot clone with `--recurse-submodules` then clone the git dependencies with the following:
```shell
git submodule update --init --recursive
```
## Setting Up Python
Start by choosing on of the following methods for setting up the Python environment (Docker is the recommended approach):
1. Conda

    You can install all the required Python dependencies with [Conda](https://docs.conda.io/en/latest/miniconda.html):
    ```shell
    conda env create -f environment.yml
    conda activate video2mesh
    ```

2. PIP - CPU Only
    ```shell
    pip install -r requirement.txt
    pip install torch==1.10.0+cpu torchvision==0.11.1+cpu -f https://download.pytorch.org/whl/cpu/torch_stable.html
    pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cpu/torch1.10/index.html
    ```

3. PIP - CUDA (11.3)
    ```shell
    pip install -r requirement.txt
    pip install torch==1.10.0+cu113 torchvision==0.11.1+cu113 -f https://download.pytorch.org/whl/cpu/torch_stable.html
    pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cuda113/torch1.10/index.html
    ```

4. Docker - CUDA (11.3)

   Either:
5. Pull (download) a pre-built image (~11.3 GB): 
   ```shell
   docker pull eight0153/video2mesh:cu116
   ```
6. Build the Docker Image:
    
   ```shell
   docker build -t eight0153/video2mesh:cu116 .
   ```
        
   **Note:** For M1 Macs you should specify amd64 as the platform:
   ```shell
   docker buildx build --platform linux/amd64 -t eight0153/video2mesh:cu116 .
   ```
   It is important to do this as not all the required packages have arm64 pre-built binaries available.

**Note**: If you either set up the environment or build the Docker image locally, you will need to 
download the weights for the depth estimation models from [Google Drive](https://drive.google.com/file/d/1lvyZZbC9NLcS8a__YPcUP7rDiIpbRpoF/view?usp=sharing) and [CloudStor](https://cloudstor.aarnet.edu.au/plus/s/lTIJF4vrvHCAI31), and place them in the folder `weights/`. 

## Setting Up C++ Environment
If you use the Docker image, you do not need to do anything for this step.
Otherwise, ensure that you have installed the following:
- CUDA Toolkit 11.6
- OpenCV 3.4.16 
  - Make sure to enable the CMake flag `-DWITH_CUDA=true`.
  
Refer to `Dockerfile` for detailed setup instructions on Ubuntu 20.04.  

## Running the Program
### Sample Dataset
You can download a sample dataset from the [TUM website](https://vision.in.tum.de/data/datasets/rgbd-dataset/download).
The sequence `fr3/walking_xyz` is a good one to start with.
Make sure you download and extract the dataset to the `data/` folder.

### Example Usage
Below is an example of how to run the program:
```shell
python Video2mesh/video2mesh.py --base_path data/rgbd_dataset_freiburg3_walking_xyz --num_frames 10 --max_depth_dist 0.1 --include_background --static_background --overwrite_ok
```

 **Note:** Creating the instance segmentation masks with a CPU only image/Python environment will be *VERY* slow. 
 It is strongly recommended that you use a GPU image/environment if possible.

### CLI Parameters
If you want help with the CLI and the options, you can either refer to the source code or view the help via:
```shell
python Video2mesh/video2mesh.py --help
```

### Docker
The Docker containers will, by default, bring up the python interpreter.
All you need to do to get the main script (or any other script) running is to append the usual command, 
minus the call to python, to the following:
```shell
docker run --rm -gpus all -v $(pwd)/data:/app/data -v $(pwd)/Video2mesh:/app/Video2mesh -t eight0153/video2mesh:cu116 
```
For example, if you wanted to run the CUDA enabled container: 
```shell
docker run --rm -gpus all -v $(pwd)/data:/app/data -v $(pwd)/Video2mesh:/app/Video2mesh -t eight0153/video2mesh:cu116 -c "import torch; print(torch.cuda.is_available())"
```

### Viewing the 3D Video
Refer to the [WebXR repo](https://github.com/eight0153/webxr3dvideo).

## Input Data Format
This program accepts datasets in three formats:
- TUM [RGB-D SLAM Dataset](https://vision.in.tum.de/data/datasets/rgbd-dataset/file_formats)
- RGB-D datasets created on an iOS device using [StrayScanner](https://apps.apple.com/nz/app/stray-scanner/id1557051662)
- The VTM format (see below.)

The above datasets are automatically converted to the VTM format if they are not already in that format.

Overall, the expected folder structure is as follows:

```
<project root>
│   ...
│
└───data
│   │
│   └───<dataset 1>
│   │   │   metadata.json
│   │   │   camera_matrix.txt
│   │   │   camera_trajectory.txt
│   │   │   rgb
│   │   │   depth
│   │
│   └───...
│   │
│   └───<dataset n>
└───...
```

Datasets should be placed in a folder inside the `data/` folder.
Generally, the number of colour frames must match the number of depth maps, masks and lines in the camera trajectory 
file.
Within each dataset folder, there should be the following 5 items:
1. The metadata in a JSON formatted file that contains the following fields:
   - `num_frames`: The number of frames in the video sequence.
   - `fps`: The framerate of the video.
   - `width`: The width of the video frames in pixels.
   - `height`: The height of the video frames in pixels.
   - `depth_scale`: A scalar that when multiplied with a depth value converts that depth value to meters.
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
   Absolute pose values are expected (i.e. all relative to world origin).
4. The colour (RGB) frames, either JPEG or PNG, in a folder with names that preserve the frames' natural ordering, e.g.:
   ```text
   rgb
   │   000001.jpg
   │   000002.jpg
   │   ...
   └───999999.jpg
   ```
5. The depth maps (either JPEG, PNG, or .raw) in a folder with names that preserve the frames' natural ordering, e.g.:
   ```text
   depth
   │   000001.jpg
   │   000002.jpg
   │   ...
   └───999999.jpg
   ```
   The depth maps are expected to be stored in a 16-bit grayscale image. The depth values should be in millimeters and increasing from the camera (i.e. depth = 0 at the camera).

## Output Format
The generated meshes are saved to a glTF formatted file.
Each mesh in the glTF file represents the mesh for all objects for a given frame.