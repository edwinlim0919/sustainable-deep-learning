#!/bin/bash
export PATH="$HOME/miniconda3/bin:$PATH"
export PATH='/users/'"${USER}"'/miniconda3/envs/intel-transformers/bin:'"$PATH"
conda init
conda activate intel-transformers
