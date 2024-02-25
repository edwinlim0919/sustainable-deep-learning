#!/usr/bin/env bash

sudo apt-get install libssl-dev -y
sudo apt-get install libz-dev -y
sudo apt-get install luarocks -y
sudo luarocks install luasocket
sudo apt install linux-tools-`uname -r` linux-tools-generic htop -y
sudo apt install libelf-dev libdw-dev systemtap-sdt-dev libunwind-dev libslang2-dev libnuma-dev libiberty-dev -y
sudo apt-get update
sudo apt-get install -y cmake
sudo apt install docker.io



mkdir -p ~/miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
rm -rf ~/miniconda3/miniconda.sh
~/miniconda3/bin/conda init bash
export PATH="$HOME/miniconda3/bin:$PATH"

conda init
conda create --name intel-transformers python=3.11
conda activate intel-transformers
conda install pip
export PATH='/users/'"${USER}"'/miniconda3/envs/intel-transformers/bin:'"$PATH"



python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
python -m pip install intel-extension-for-pytorch
python -m pip install oneccl_bind_pt --extra-index-url https://pytorch-extension.intel.com/release-whl/stable/cpu/us/
pip install -r requirements.txt

pip install -r neural-speed/requirements.txt
pip install ./neural-speed/



echo "conda activate intel-transformers" >> ~/.bashrc
