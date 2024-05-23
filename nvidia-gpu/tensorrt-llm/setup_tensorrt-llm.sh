#!/usr/bin/env bash


# Install OpenMPI
sudo apt-get update
sudo apt-get install openmpi-bin openmpi-common libopenmpi-dev


# Install NVIDIA drivers
sudo apt-get update
sudo apt-get install nvidia-driver-535
sudo nvidia-persistenced  --user root
nvidia-smi


# Create conda environment with correct Python version
conda create --name tensorrt-llm python=3.10
conda activate tensorrt-llm
pip3 install -r requirements.txt


# Setting up environment for NVIDIA Docker containers
# Setup instructions from https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
&& curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
  sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
  sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit

sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker


# Building from source code
#sudo apt-get update
#sudo apt-get -y install git git-lfs
#
#git clone https://github.com/NVIDIA/TensorRT-LLM.git
#cd TensorRT-LLM
#git submodule update --init --recursive
#git lfs pull


# Installing through pip
sudo docker run --rm --runtime=nvidia --gpus all --entrypoint /bin/bash -it nvidia/cuda:12.2.0-devel-ubuntu22.04
apt-get update
apt-get -y install python3.10 python3-pip openmpi-bin libopenmpi-dev git
pip3 install tensorrt_llm -U --pre --extra-index-url https://pypi.nvidia.com
python3 -c "import tensorrt_llm"

git clone https://github.com/NVIDIA/TensorRT-LLM.git
cd TensorRT-LLM


# Install various dependencies for TRT-LLM
apt-get update
apt-get install wget
