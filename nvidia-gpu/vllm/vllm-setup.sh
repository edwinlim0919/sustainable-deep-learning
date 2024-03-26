#!/usr/bin/env bash

#conda create -n vllm-env python=3.9 -y
conda create --name  vllm-env python=3.9
conda activate vllm-env

pip install vllm
