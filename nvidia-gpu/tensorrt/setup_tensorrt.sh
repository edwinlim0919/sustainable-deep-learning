#!/usr/bin/env bash

# Install NVIDIA drivers
sudo apt-get update
sudo apt-get install -y nvidia-driver-535
sudo nvidia-persistenced  --user root
nvidia-smi


# Create conda environment with correct Python version
conda create --name tensorrt python=3.10
conda activate tensorrt


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


# Installing through pip and NVIDIA docker containers
sudo docker run --gpus all -it --rm nvcr.io/nvidia/pytorch:24.04-py3


#python -m pip install torch torch-tensorrt tensorrt --extra-index-url https://download.pytorch.org/whl/cu122
