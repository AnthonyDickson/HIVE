FROM nvidia/cuda:11.6.0-devel-ubuntu20.04

RUN apt update && \
    DEBIAN_FRONTEND="noninteractive" apt install -y --no-install-recommends \
    python3 python3-dev python3-pip wget unzip make cmake ninja-build gcc g++ libgl1-mesa-dev libglib2.0-0 \
    # COLMAP Dependencies
    build-essential \
    git libboost-program-options-dev libboost-filesystem-dev libboost-graph-dev  \
    libboost-regex-dev libboost-system-dev libboost-test-dev libeigen3-dev libsuitesparse-dev libfreeimage-dev  \
    libgoogle-glog-dev libgflags-dev libglew-dev qtbase5-dev libqt5opengl5-dev libcgal-dev libcgal-qt5-dev \
    # I think these are for FFMPEG.
    libavcodec-dev libavformat-dev libavutil-dev libswscale-dev libgtk2.0-dev pkg-config \
    # Packages for Ceres Solver
    libatlas-base-dev libsuitesparse-dev libeigen3-dev  \
    # BundleFusion
    libncurses5-dev libncursesw5-dev && \
    apt-get -y autoremove && apt-get -y clean && apt-get -y autoclean && \
    rm -rf /var/lib/apt/lists/*
# COLMAP
## Ceres solver
RUN cd / && \
    git clone https://ceres-solver.googlesource.com/ceres-solver && \
    cd ceres-solver && \
    git checkout $(git describe --tags) && \
    mkdir build && \
    cd build && \
    cmake .. -DBUILD_TESTING=OFF -DBUILD_EXAMPLES=OFF && \
    make -j 8 && \
    make install

## Install COLMAP from source
# Download and install OpenCV
ENV OPENCV_VERSION=3.4.16

RUN cd / && \
    wget -O opencv.zip https://github.com/opencv/opencv/archive/refs/tags/${OPENCV_VERSION}.zip && \
    unzip opencv.zip && \
    mkdir -p opencv-build &&  \
    cd opencv-build && \
    cmake -D WITH_CUDA=ON \
    -D BUILD_EXAMPLES=OFF -D BUILD_opencv_apps=OFF -D BUILD_DOCS=OFF -D BUILD_PERF_TESTS=OFF -D BUILD_TESTS=OFF ../opencv-${OPENCV_VERSION} && \
    cmake --build . -- -j 8 && \
    make install && \
    cd .. && \
    rm opencv.zip && \
    rm -rf opencv-${OPENCV_VERSION} && \
    rm -rf opencv-build


ARG BUNDLE_FUSION_FOLDER=bundle_fusion
ENV BUNDLE_FUSION_PATH=/${BUNDLE_FUSION_FOLDER}
ENV BUNDLE_FUSION_BIN=/${BUNDLE_FUSION_FOLDER}/build/bundle_fusion_example

RUN git clone https://github.com/eight0153/BundleFusion_Ubuntu_Pangolin.git ${BUNDLE_FUSION_FOLDER} && \
    cd ${BUNDLE_FUSION_FOLDER} && \
    git checkout 2b9c1df && \
    mkdir build && \
    cd build && \
    cmake -DVISUALIZATION=OFF .. && \
    make -j8

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir torch==1.10.2+cu113 torchvision==0.11.3+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html
RUN pip install --no-cache-dir detectron2 -f \
  https://dl.fbaipublicfiles.com/detectron2/wheels/cu113/torch1.10/index.html

WORKDIR /app

# Download instance segmentation weights and weights for depth estimation model (NYU only) so
# they do not need to be downloaded each time you run the container.
ADD scripts scripts

ENV COLMAP_VOCAB_PATH=/root/.cache/colmap
RUN python3 scripts/download_detectron2_weights.py && \
    # AdaBins Weights
    python3 scripts/download_adabins_basemodel.py &&  \
    mkdir -p ${COLMAP_VOCAB_PATH} && \
    wget https://demuc.de/colmap/vocab_tree_flickr100K_words256K.bin -O ${COLMAP_VOCAB_PATH}/vocab.bin

ENV WEIGHTS_PATH=/root/.cache/pretrained
COPY weights/AdaBins_nyu.pt ${WEIGHTS_PATH}/AdaBins_nyu.pt
COPY weights/res101.pth ${WEIGHTS_PATH}/res101.pth

ENTRYPOINT ["python3"]
