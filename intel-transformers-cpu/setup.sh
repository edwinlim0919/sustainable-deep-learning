#!/usr/bin/env bash
git submodule init ../neural-speed
git submodule update ../neural-speed
git submodule init pcm
git submodule update pcm

CURR_DIR=$(pwd)
cd pcm/
git submodule update --init --recursive
mkdir build
cd build
cmake ..
cmake --build .
cd "$CURR_DIR"
sudo apt install pcm


conda create --name intel-transformers python=3.11
conda activate intel-transformers
conda install pip
export PATH='/users/'"${USER}"'/miniconda3/envs/intel-transformers/bin:'"$PATH"


python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
python -m pip install intel-extension-for-pytorch
python -m pip install oneccl_bind_pt --extra-index-url https://pytorch-extension.intel.com/release-whl/stable/cpu/us/
pip install -r requirements.txt


pip install -r ../neural-speed/requirements.txt
pip install ./../neural-speed/
