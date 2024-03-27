#!/usr/bin/env bash

conda activate tensorrt-llm
huggingface-cli login

# TODO: Currently only does Llama2-7B
CURR_DIR=$(pwd)
MODEL_NAME="meta-llama/Llama-2-7b-chat-hf"
MODEL_WEIGHT_DIR="${CURR_DIR}/${MODEL_NAME}_model"
MODEL_TOKENIZER_DIR="${CURR_DIR}/${MODEL_NAME}_tokenizer"
python3 download_hf_weights.py --model-name $MODEL_NAME

cd TensorRT-LLM/examples/llama
pip install -r requirements.txt

python3 convert_checkpoint.py --model_dir $MODEL_WEIGHT_DIR --output_dir ./tllm_checkpoint_1gpu_fp16 --dtype float16

# To run:
# python3 ../run.py --max_output_len=50 --tokenizer_dir ../../../meta-llama/Llama-2-7b-chat-hf_tokenizer --engine_dir=./tllm_checkpoint_1gpu_fp16
