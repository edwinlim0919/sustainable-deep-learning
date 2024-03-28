#!/usr/bin/env bash

# Installing the NVIDIA container toolkit
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
&& curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
  sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
  sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
#sudo sed -i -e '/experimental/ s/^#//g' /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit

sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker

sudo apt-get update
sudo apt install openmpi-bin openmpi-common libopenmpi-dev

sudo apt-get update
sudo apt-get install git-lfs
git lfs install

#nvidia-ctk runtime configure --runtime=docker --config=$HOME/.config/docker/daemon.json
#sudo systemctl restart docker
#sudo nvidia-ctk config --set nvidia-container-cli.no-cgroups --in-place

#sudo nvidia-ctk runtime configure --runtime=containerd
#sudo systemctl restart containerd

conda create --name tensorrt-llm python=3.10
conda activate tensorrt-llm
conda install mpi4py

# TensorRT installation
sudo docker run --rm --runtime=nvidia --gpus all --entrypoint /bin/bash -it nvidia/cuda:12.1.0-devel-ubuntu22.04
sudo apt-get update
sudo apt-get -y install python3.10 python3-pip openmpi-bin libopenmpi-dev
pip3 install tensorrt_llm -U --pre --extra-index-url https://pypi.nvidia.com

git clone git@github.com:NVIDIA/TensorRT-LLM.git
#git clone git@github.com:huggingface/transformers.git


# TensorRT-LLM Backend
#git@github.com:triton-inference-server/tensorrtllm_backend.git
#BASE_IMAGE=nvcr.io/nvidia/tritonserver:24.01-py3-min
#TRT_VERSION=9.2.0.5
#TRT_URL_x86=https://developer.nvidia.com/downloads/compute/machine-learning/tensorrt/9.2.0/tensorrt-9.2.0.5.linux.x86_64-gnu.cuda-12.2.tar.gz
#TRT_URL_ARM=https://developer.nvidia.com/downloads/compute/machine-learning/tensorrt/9.2.0/tensorrt-9.2.0.5.Ubuntu-22.04.aarch64-gnu.cuda-12.2.tar.gz
#
#sudo docker build -t trtllm_base \
#--build-arg BASE_IMAGE="${BASE_IMAGE}" \
#--build-arg TRT_VER="${TRT_VERSION}" \
#--build-arg RELEASE_URL_TRT_x86="${TRT_URL_x86}" \
#--build-arg RELEASE_URL_TRT_ARM="${TRT_URL_ARM}" \
#-f dockerfile/Dockerfile.triton.trt_llm_backend .
