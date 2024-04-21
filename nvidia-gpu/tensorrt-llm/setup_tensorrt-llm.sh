#!/usr/bin/env bash

# TODO: This installation may conflict with the official nvidia installer installation
# Installing the NVIDIA container toolkit
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
&& curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
  sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
  sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit

sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker

#sudo apt-get update
#sudo apt install openmpi-bin openmpi-common libopenmpi-dev

sudo apt-get update
sudo apt-get -y install git git-lfs
git lfs install

conda create --name tensorrt-llm python=3.10
conda activate tensorrt-llm
pip install -r requirements.txt
#conda install mpi4py

git clone https://github.com/NVIDIA/TensorRT-LLM.git
cd TensorRT-LLM
git submodule update --init --recursive
git lfs pull
sudo make -C docker release_build

sudo nvidia-persistenced --user root
# Starting the container
#sudo make -C docker release_run

# TODO: check if nvidia-smi can run
#       if conflict, then run below command
sudo apt-get --purge remove "*nvidia*"
