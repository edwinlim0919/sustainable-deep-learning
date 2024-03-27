# Sustainable Deep Learning
Scripting for running deep learning sustainability experiments on various hardware platforms

# Base Setup
First, clone the repository
```bash
cd /dev/shm
git clone git@github.com:edwinlim0919/sustainable-deep-learning.git
cd sustainable-deep-learning
```

Next, set up some bash environment stuff.
You may need to log out of the node and log back in to see env changes take affect.
```bash
yes | source ./env.sh && yes | source ./setup.sh
```

# OpenVINO Setup
```bash
cd /dev/shm/sustainable-deep-learning/intel-cpu/OpenVINO
source ./setup-openvino.sh
```

# NVIDIA GPU Setup (CUDA 12.1)
```bash
# This will also reboot your node and you will likely need to re-clone this repo in /dev/shm
cd /dev/shm/sustainable-deep-learning/nvidia-gpu
sudo ./remove-nouveau.sh

# This script will reboot your node as well (example provided below)
# Figure out which version of CUDA 12.1 to use for your machine from here: https://developer.nvidia.com/cuda-12-1-0-download-archive
cd /dev/shm/sustainable-deep-learning/nvidia-gpu
sudo ./nvidia-setup.sh https://developer.download.nvidia.com/compute/cuda/12.1.0/local_installers/cuda_12.1.0_530.30.02_linux.run cuda_12.1.0_530.30.02_linux.run
```

## vLLM Setup
conda create --name  vllm-env python=3.9
conda activate vllm-env
pip install vllm
