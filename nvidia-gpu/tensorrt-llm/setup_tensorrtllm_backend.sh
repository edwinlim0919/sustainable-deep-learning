#!/usr/bin/env bash

# TensorRT-LLM Backend
git clone git@github.com:triton-inference-server/tensorrtllm_backend.git
cd tensorrtllm_backend

# 9.3.0.1
BASE_IMAGE=nvcr.io/nvidia/tritonserver:24.01-py3-min
TRT_VERSION=9.2.0.5
TRT_URL_x86=https://developer.nvidia.com/downloads/compute/machine-learning/tensorrt/9.2.0/tensorrt-9.2.0.5.linux.x86_64-gnu.cuda-12.2.tar.gz
TRT_URL_ARM=https://developer.nvidia.com/downloads/compute/machine-learning/tensorrt/9.2.0/tensorrt-9.2.0.5.Ubuntu-22.04.aarch64-gnu.cuda-12.2.tar.gz
sudo docker build -t trtllm_base \
	--build-arg BASE_IMAGE="${BASE_IMAGE}" \
	--build-arg TRT_VER="${TRT_VERSION}" \
	--build-arg RELEASE_URL_TRT_x86="${TRT_URL_x86}" \
	--build-arg RELEASE_URL_TRT_ARM="${TRT_URL_ARM}" \
	-f dockerfile/Dockerfile.triton.trt_llm_backend .

cd ../
git clone git@github.com:triton-inference-server/server.git
conda create --name tensorrtllm_backend python=3.10
conda activate tensorrtllm_backend
pip install -r requirements.txt

# Triton inference server
TRTLLM_BASE_IMAGE=trtllm_base
TENSORRTLLM_BACKEND_REPO_TAG=v0.7.2
PYTHON_BACKEND_REPO_TAG=r24.01

PYPATH=$(which python3)
cd server/
sudo $PYPATH build.py -v --no-container-interactive --enable-logging --enable-stats --enable-tracing \
	--enable-metrics --enable-gpu-metrics --enable-cpu-metrics \
	--filesystem=gcs --filesystem=s3 --filesystem=azure_storage \
	--endpoint=http --endpoint=grpc --endpoint=sagemaker --endpoint=vertex-ai \
	--backend=ensemble --enable-gpu --endpoint=http --endpoint=grpc \
	--no-container-pull \
	--image=base,${TRTLLM_BASE_IMAGE} \
	--backend=tensorrtllm:${TENSORRTLLM_BACKEND_REPO_TAG} \
	--backend=python:${PYTHON_BACKEND_REPO_TAG}

# Initialize TensorRT-LLM submodule
cd ../tensorrtllm_backend
sudo apt-get update
sudo apt-get install git-lfs
git submodule update --init --recursive
git lfs install
git lfs pull

cd tensorrt_llm
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
&& curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
  sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
  sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
sudo apt-get update
sudo apt install openmpi-bin openmpi-common libopenmpi-dev

sudo nvidia-persistenced --user root
sudo docker run --rm --runtime=nvidia --gpus all --entrypoint /bin/bash -it nvidia/cuda:12.2.0-devel-ubuntu22.04
exit
sudo apt-get update
sudo apt-get -y install python3.10 python3-pip openmpi-bin libopenmpi-dev
pip3 install tensorrt_llm -U --pre --extra-index-url https://pypi.nvidia.com

cd ../
mkdir triton_model_repo
