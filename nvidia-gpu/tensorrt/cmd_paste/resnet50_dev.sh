#!/usr/bin/env bash


### MISCELLANEOUS HELPER COMMANDS
# replace docker container id in this cmd_paste with the current one
sed -i "s/9c529806c267/9c529806c267/g" resnet50_dev.sh

# flushing GPU memory
sudo fuser -v /dev/nvidia*
sudo kill -9 <PID>

# disable MIG if Pytorch cannot find CUDA
nvidia-smi
sudo nvidia-smi -mig 0
nvidia-smi


# RESNET50 SETUP COMMANDS
sudo docker cp ../benchmarking/resnet50_example.py 9c529806c267:/workspace/resnet50_example.py
