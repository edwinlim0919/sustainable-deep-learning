#!/usr/bin/env bash

git clone --recursive https://github.com/apache/tvm tvm
sudo apt-get update
sudo apt-get install -y python3 python3-dev python3-setuptools gcc libtinfo-dev zlib1g-dev build-essential cmake libedit-dev libxml2-dev
sudo apt-get install llvm

# Install Intel OneDNN
#git clone https://github.com/oneapi-src/oneDNN.git
#cd oneDNN
#mkdir build
#cd build
#cmake ..
#make
#sudo make install
#cd ../../

wget https://registrationcenter-download.intel.com/akdlm/IRC_NAS/163da6e4-56eb-4948-aba3-debcec61c064/l_BaseKit_p_2024.0.1.46_offline.sh
sudo sh ./l_BaseKit_p_2024.0.1.46_offline.sh

cd tvm
mkdir build
cp cmake/config.cmake build/config.cmake

# set(USE_LLVM ON)
sed -i 's/set(USE_LLVM OFF)/set(USE_LLVM ON)/g' build/config.cmake

# TODO: Currently errors with using OneDNN, need to look into this
# set(USE_DNNL ON)
#sed -i 's/set(USE_DNNL OFF)/set(USE_DNNL ON)/g' build/config.cmake

# TODO: Intel OneDNN already usese OpenMP, so not sure if I should explicitly turn OpenMP on
# set(USE_OPENMP intel)
#sed -i 's/set(USE_OPENMP none)/set(USE_OPENMP intel)/g' build/config.cmake

# TODO: For 4th gen Xeon platforms, turn on Intel AMX Instructions
# set(USE_AMX ON)
#sed -i 's/set(USE_AMX OFF)/set(USE_AMX ON)/g' build/config.cmake

cd build
cmake ..
make -j4
