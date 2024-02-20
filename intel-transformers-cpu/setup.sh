#!/usr/bin/env bash
conda create --name intel-transformers python=3.11
conda activate intel-transformers
conda install pip
export PATH='/users/'"${USER}"'/miniconda3/envs/intel-transformers/bin:'"$PATH"
#conda deactivate


python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
python -m pip install intel-extension-for-pytorch
python -m pip install oneccl_bind_pt --extra-index-url https://pytorch-extension.intel.com/release-whl/stable/cpu/us/

pip install -r requirements.txt


pip install -r ../neural-speed/requirements.txt
pip install ./../neural-speed/
