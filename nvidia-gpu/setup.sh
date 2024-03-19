conda create --name nvidia-gpu
conda activate nvidia-gpu
conda install pip
export PATH='/users/'"${USER}"'/miniconda3/envs/nvidia-gpu/bin:'"$PATH"

pip install torch
pip install numpy

git clone https://github.com/flashinfer-ai/flashinfer.git --recursive
cd flashinfer/python && pip install -e .
