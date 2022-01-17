FROM pytorch/pytorch:1.10.0-cuda11.3-cudnn8-runtime

RUN apt update && \
    DEBIAN_FRONTEND="noninteractive" apt install -y --no-install-recommends \
    python3 python3-dev python3-pip make cmake ninja-build gcc g++ libgl1-mesa-dev libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
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