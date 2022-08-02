FROM nvidia/cuda:11.6.1-devel-ubuntu20.04

RUN apt update && \
    DEBIAN_FRONTEND="noninteractive" apt install -y --no-install-recommends \
    python3 python3-dev python3-pip wget unzip make cmake ninja-build gcc g++ libgl1-mesa-dev libglib2.0-0 \
    # COLMAP Dependencies
    build-essential \
    git libboost-program-options-dev libboost-filesystem-dev libboost-graph-dev  \
    libboost-regex-dev libboost-system-dev libboost-test-dev libeigen3-dev libsuitesparse-dev libfreeimage-dev  \
    libgoogle-glog-dev libgflags-dev libglew-dev qtbase5-dev libqt5opengl5-dev libcgal-dev libcgal-qt5-dev \
    # I think these are for FFMPEG.
    ffmpeg libavcodec-dev libavformat-dev libavutil-dev libswscale-dev libgtk2.0-dev pkg-config \
    # Packages for Ceres Solver
    libatlas-base-dev libsuitesparse-dev libeigen3-dev  \
    # BundleFusion
    libncurses5-dev libncursesw5-dev && \
    apt -y autoremove && \
    apt -y clean && \
    apt -y autoclean && \
    rm -rf /var/lib/apt/lists/*

# COLMAP
## Ceres solver
RUN cd / && \
    git clone --depth 1 --branch 2.1.0 https://ceres-solver.googlesource.com/ceres-solver && \
    cd ceres-solver && \
    git checkout $(git describe --tags) && \
    mkdir build && \
    cd build && \
    cmake .. -DBUILD_TESTING=OFF -DBUILD_EXAMPLES=OFF && \
    make -j 8 && \
    make install && \
    cd .. && \
    rm -rf ceres-solver

## Download and install OpenCV
RUN cd / && \
    git clone --depth 1 --branch 3.4.16 https://github.com/opencv/opencv.git && \
    cd opencv && \
    mkdir -p build &&  \
    cd build && \
    cmake -D WITH_CUDA=ON \
    -D BUILD_EXAMPLES=OFF -D BUILD_opencv_apps=OFF -D BUILD_DOCS=OFF -D BUILD_PERF_TESTS=OFF -D BUILD_TESTS=OFF .. && \
    cmake --build . -- -j 8 && \
    make install && \
    cd / && \
    rm -rf opencv

## Install COLMAP from source
RUN cd / && \
    git clone --depth 1 --branch 3.7 https://github.com/colmap/colmap.git && \
    cd colmap && \
    mkdir build && \
    cd build && \
    cmake .. && \
    make -j 8 && \
    make install && \
    cd / && \
    rm -rf colmap

ARG BUNDLE_FUSION_FOLDER=bundle_fusion
ENV BUNDLE_FUSION_PATH=/${BUNDLE_FUSION_FOLDER}
ENV BUNDLE_FUSION_BIN=/${BUNDLE_FUSION_FOLDER}/build/bundle_fusion_example

RUN git clone https://github.com/AnthonyDickson/BundleFusion_Ubuntu_Pangolin.git ${BUNDLE_FUSION_FOLDER} && \
    cd ${BUNDLE_FUSION_FOLDER} && \
    git checkout 2b9c1df && \
    mkdir build && \
    cd build && \
    cmake -DVISUALIZATION=OFF .. && \
    make -j8

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

ENV WEIGHTS_PATH=/root/.cache/pretrained
ENV COLMAP_VOCAB_PATH=/root/.cache/colmap

COPY scripts/download_weights.py download_weights.py
# The first line downloads the weights for the MC depth estimation model, the AdaBins base model and Detectron2.
RUN python3 download_weights.py && \
    # Weights for the AdaBins decoder portion of the model.
    gdown https://drive.google.com/uc?id=1lvyZZbC9NLcS8a__YPcUP7rDiIpbRpoF -O ${WEIGHTS_PATH}/AdaBins_nyu.pt && \
    # FlowNet2 weights
    gdown https://drive.google.com/uc?id=1hF8vS6YeHkx3j2pfCeQqqZGwA_PJq_Da -O ${WEIGHTS_PATH}/flownet2.pth && \
    # DPT weights
    wget https://github.com/intel-isl/DPT/releases/download/1_0/dpt_hybrid_nyu-2ce69ec7.pt -O ${WEIGHTS_PATH}/dpt_hybrid_nyu.pt && \
    # COLMAP vocab file.
    mkdir -p ${COLMAP_VOCAB_PATH} && \
    wget https://demuc.de/colmap/vocab_tree_flickr100K_words256K.bin -O ${COLMAP_VOCAB_PATH}/vocab.bin && \
    rm download_weights.py

# Weights for LeReS depth estimation model.
COPY weights/res101.pth ${WEIGHTS_PATH}/res101.pth

WORKDIR /app
CMD ["python3"]