#!/usr/bin/env bash

#pip install -r requirements.txt
mkdir -p ~/miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
rm -rf ~/miniconda3/miniconda.sh

~/miniconda3/bin/conda init bash


#conda create --name neural-speed-env
#conda activate neural-speed-env
#export PATH='/users/'"${USER}"'/miniconda3/envs/neural-speed-env/bin:'"$PATH"
#conda install pip
#pip install -r neural-speed/requirements.txt
#pip install ./neural-speed/
#conda deactivate
#
#conda create --name llama-cpp
#conda activate llama-cpp
#export PATH='/users/'"${USER}"'/miniconda3/envs/llama-cpp/bin:'"$PATH"
#conda install pip
#pip install -U pyllama transformers
#pip install -r requirements.txt
#conda deactivate

conda create --name intel-transformers python=3.11
conda activate intel-transformers
conda install pip
export PATH='/users/'"${USER}"'/miniconda3/envs/intel-transformers/bin:'"$PATH"
#pip install -r neural-speed/requirements.txt
#pip install ./neural-speed/
conda deactivate
