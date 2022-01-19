FROM nvidia/cuda:11.3.1-devel-ubuntu20.04

RUN apt update && \
    DEBIAN_FRONTEND="noninteractive" apt install -y --no-install-recommends \
    python3 python3-dev python3-pip make cmake ninja-build gcc g++ libgl1-mesa-dev libglib2.0-0 \
    # COLMAP Dependencies
    build-essential \
    git libboost-program-options-dev libboost-filesystem-dev libboost-graph-dev  \
    libboost-regex-dev libboost-system-dev libboost-test-dev libeigen3-dev libsuitesparse-dev libfreeimage-dev  \
    libgoogle-glog-dev libgflags-dev libglew-dev qtbase5-dev libqt5opengl5-dev libcgal-dev libcgal-qt5-dev \
    # Packages for Ceres Solver
    libatlas-base-dev libsuitesparse-dev libeigen3-dev && \
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
RUN cd / && \
    git clone https://github.com/colmap/colmap.git && \
    cd colmap && \
    git checkout dev && \
    mkdir build && \
    cd build && \
    cmake .. && \
    make -j 8 && \
    make install


COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir torch==1.10.1+cu113 torchvision==0.11.2+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html
RUN pip install --no-cache-dir detectron2 -f \
  https://dl.fbaipublicfiles.com/detectron2/wheels/cu113/torch1.10/index.html

WORKDIR /app

# Download instance segmentation weights and weights for depth estimation model (NYU only) so
# they do not need to be downloaded each time you run the container.
ADD scripts scripts
ARG WEIGHTS_PATH=/root/.cache/pretrained
RUN python3 scripts/download_detectron2_weights.py && \
    python3 scripts/download_adabins_basemodel.py && \
    gdown --id 1lvyZZbC9NLcS8a__YPcUP7rDiIpbRpoF && mkdir -p ${WEIGHTS_PATH} && mv AdaBins_nyu.pt ${WEIGHTS_PATH}/AdaBins_nyu.pt

ENTRYPOINT ["python3"]