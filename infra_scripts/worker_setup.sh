#!/usr/bin/env bash

sudo apt-get install libssl-dev -y
sudo apt-get install libz-dev -y
sudo apt-get install luarocks -y
sudo luarocks install luasocket
sudo apt install linux-tools-`uname -r` linux-tools-generic htop -y
sudo apt install libelf-dev libdw-dev systemtap-sdt-dev libunwind-dev libslang2-dev libnuma-dev libiberty-dev -y

sudo apt-get update
sudo apt-get install -y cmake


mkdir -p ~/miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
rm -rf ~/miniconda3/miniconda.sh

~/miniconda3/bin/conda init bash


sudo apt install docker.io
