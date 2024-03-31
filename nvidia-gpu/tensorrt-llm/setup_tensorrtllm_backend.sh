#!/usr/bin/env bash

# TensorRT-LLM Backend
conda create --name tensorrtllm_backend python=3.10
conda activate tensorrtllm_backend
PYPATH=$(which python3)

git clone git@github.com:triton-inference-server/tensorrtllm_backend.git
cd tensorrtllm_backend
git lfs install
git submodule update --init --recursive
DOCKER_BUILDKIT=1 docker build -t triton_trt_llm -f dockerfile/Dockerfile.trt_llm_backend .

git submodule update --init --recursive
git lfs install
git lfs pull

sudo apt-get update
sudo apt-get install cmake

#sudo groupadd docker
#sudo usermod -aG docker $USER
#newgrp docker
#conda activate tensorrtllm_backend

cd tensorrt_llm
sudo docker run --rm --runtime=nvidia --gpus all --entrypoint /bin/bash -it nvidia/cuda:12.2.0-devel-ubuntu22.04
sudo apt-get update
sudo apt-get -y install python3.10 python3-pip openmpi-bin libopenmpi-dev
pip3 install tensorrt_llm -U --pre --extra-index-url https://pypi.nvidia.com

#sudo bash docker/common/install_cmake.sh
#echo 'export PATH=$PATH:/usr/local/cmake/bin' >> ~/.bashrc
#source ~/.bashrc
#export PATH=/usr/local/cmake/bin:$PATH
#conda install mpi4py
#sudo $PYPATH ./scripts/build_wheel.py --trt_root="/usr/local/tensorrt"
