#!/usr/bin/env bash

# Intel OneAPI Installation
# TODO: This is specific to c6420 node
wget https://registrationcenter-download.intel.com/akdlm/IRC_NAS/163da6e4-56eb-4948-aba3-debcec61c064/l_BaseKit_p_2024.0.1.46_offline.sh
sudo sh ./l_BaseKit_p_2024.0.1.46_offline.sh

# OpenVINO Installation
#sudo apt-key add GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB
#echo "deb https://apt.repos.intel.com/openvino/2024 ubuntu22 main" | sudo tee /etc/apt/sources.list.d/intel-openvino-2024.list
#sudo apt update
#apt-cache search openvino
#sudo apt install openvino-2024.0.0

sudo apt-get update
sudo apt-get install ffmpeg libsm6 libxext6  -y

# TODO: Don't think we need both of these installs
conda create --name openvino python=3.11
conda activate openvino
python -m pip install --upgrade pip
#pip install openvino==2024.0.0
pip install -r requirements.txt

#python3 -m venv openvino_env
#source openvino_env/bin/activate
#python -m pip install --upgrade pip
#pip install openvino==2024.0.0
