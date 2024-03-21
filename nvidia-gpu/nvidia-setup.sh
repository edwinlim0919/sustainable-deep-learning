#!/usr/bin/env bash

# Arguments
#   1: link to CUDA installer from https://developer.nvidia.com/cuda-12-1-0-download-archive
#   2: filename of .run installer

wget $1
sudo sh $2

# TODO: not sure if explicit command line reboot works, may have to do reboot from cloudlab
sudo reboot
