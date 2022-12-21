# Load base image from Nvidia
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04
# --------- Stage 1 image building  --------
# pull ubunut base
# FROM nvidia/cuda:11.2.0-base-ubuntu20.04 as devel
# Copy wheel files and requirments.txt
COPY ./libs/wheel_files ./requirements.txt /
# For change nvidia-key
# RUN rm /etc/apt/sources.list.d/cuda.list && \
#     apt-key del 7fa2af80
# Install package for stage - 1 image
RUN apt update && \
    DEBIAN_FRONTEND=noninteractive apt install --no-install-recommends -y \
    build-essential libssl-dev libffi-dev libxml2-dev libxslt1-dev zlib1g-dev libgl1 software-properties-common wget && \
    add-apt-repository -y ppa:deadsnakes/ppa && \
    apt install --no-install-recommends -y python3.8 python3.8-dev python3-pip python3-setuptools python3-distutils && \
    # For change nvidia-key
    wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pin && \
    mv cuda-ubuntu2004.pin /etc/apt/preferences.d/cuda-repository-pin-600 &&\
    apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/3bf863cc.pub && \
    apt clean && rm -rf /var/lib/apt/lists/* && \
    pip install --no-cache-dir -r /requirements.txt && \
    pip install --no-cache-dir /torch-1.12.1+cu116-cp39-cp39-linux_x86_64.whl && \
    pip install --no-cache-dir /torchvision-0.13.1+cu116-cp39-cp39-linux_x86_64.whl && \
    ldconfig && \
    pip install --no-cache-dir /tensorflow_gpu-2.6.1-cp39-cp39-manylinux2010_x86_64.whl && \


# create a work directory
WORKDIR /codebase

# copy project to work directory
COPY ./codebase /codebase/

# we aways need to run run_server.py
CMD python3 inference.py