#!/usr/bin/env bash

conda create --name tensorrt python=3.10
conda activate tensorrt

python -m pip install torch torch-tensorrt tensorrt --extra-index-url https://download.pytorch.org/whl/cu122
