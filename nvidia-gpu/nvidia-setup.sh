#!/usr/bin/env bash

# Arguments
#   1: link to CUDA installer from https://developer.nvidia.com/cuda-12-1-0-download-archive
#   2: filename of .run installer
#
# Example Usage
#   sudo ./nvidia-setup.sh https://developer.download.nvidia.com/compute/cuda/12.1.0/local_installers/cuda_12.1.0_530.30.02_linux.run cuda_12.1.0_530.30.02_linux.run

wget $1
sudo sh $2
sudo apt install nvidia-cuda-toolkit

# TODO: not sure if explicit command line reboot works, may have to do reboot from cloudlab
sudo reboot
