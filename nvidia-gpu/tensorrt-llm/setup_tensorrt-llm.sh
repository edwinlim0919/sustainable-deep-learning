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

#nvidia-ctk runtime configure --runtime=docker --config=$HOME/.config/docker/daemon.json
#sudo systemctl restart docker
#sudo nvidia-ctk config --set nvidia-container-cli.no-cgroups --in-place
#
#sudo nvidia-ctk runtime configure --runtime=containerd
#sudo systemctl restart containerd



# TensorRT installation
sudo docker run --rm --runtime=nvidia --gpus all --entrypoint /bin/bash -it nvidia/cuda:12.1.0-devel-ubuntu22.04
sudo apt-get update
sudo apt-get -y install python3.10 python3-pip openmpi-bin libopenmpi-dev
pip3 install tensorrt_llm -U --pre --extra-index-url https://pypi.nvidia.com

