#!/usr/bin/env bash

git clone https://github.com/dusty-nv/jetson-containers
bash jetson-containers/install.sh

pip install -U "huggingface_hub[cli]"
