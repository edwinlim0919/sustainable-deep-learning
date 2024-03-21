#!/usr/bin/env bash
#
# Example Usage
#   sudo ./setup.sh NVIDIA-Linux-x86_64-535.161.08.run

# TODO make sure that NVIDIA driver installer is in this directory

sudo sh $1

sudo apt install nvidia-cuda-toolkit

conda create --name nvidia-gpu python=3.11
conda activate nvidia-gpu
conda install pip
export PATH='/users/'"${USER}"'/miniconda3/envs/nvidia-gpu/bin:'"$PATH"

#python3 -m pip install --pre -U -f https://mlc.ai/wheels mlc-ai-nightly-cu122

#sudo sh $1
#
#conda create --name nvidia-gpu python=3.11
#conda activate nvidia-gpu
#conda install pip
#export PATH='/users/'"${USER}"'/miniconda3/envs/nvidia-gpu/bin:'"$PATH"

pip install torch==2.2
pip install numpy==1.26.4
pip install flashinfer -i https://flashinfer.ai/whl/cu121/torch2.2/

#git clone https://github.com/flashinfer-ai/flashinfer.git --recursive
#cd flashinfer/python && pip install -e .
