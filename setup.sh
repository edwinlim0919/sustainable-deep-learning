#!/usr/bin/env bash

mkdir -p ~/miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
rm -rf ~/miniconda3/miniconda.sh

~/miniconda3/bin/conda init bash


#conda create --name intel-transformers python=3.11
#conda activate intel-transformers
#conda install pip
#export PATH='/users/'"${USER}"'/miniconda3/envs/intel-transformers/bin:'"$PATH"
#conda deactivate
git submodule init neural-speed
git submodule update neural-speed
