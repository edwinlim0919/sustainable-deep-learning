#!/usr/bin/env bash

conda create -n vllm-env python=3.9 -y
conda activate vllm-env

pip install vllm
