# TensorRT-LLM standalone

sed -i "s/b209d39f9c48/b209d39f9c48/g" cmd_paste_gpt2_380M_v10032gb.sh # replace docker container id in cmd_paste.sh with the current one

# Flushing GPU memory
sudo fuser -v /dev/nvidia*
sudo kill -9 <PID>

# container "/app/tensorrt_llm/examples/gpt"
pip install -r requirements.txt
rm -rf gpt2-medium && git clone https://huggingface.co/gpt2-medium gpt2-medium
pushd gpt2-medium && rm pytorch_model.bin model.safetensors && wget -q https://huggingface.co/gpt2-medium/resolve/main/pytorch_model.bin && popd

# 1-gpu fp16 nowq
python3 convert_checkpoint.py --model_dir gpt2-medium --dtype float16 --output_dir gpt2-medium/trt_ckpt/fp16/1-gpu/
trtllm-build --checkpoint_dir gpt2-medium/trt_ckpt/fp16/1-gpu/ --gemm_plugin float16 --output_dir gpt2-medium/trt_engines/fp16/1-gpu-1-batch/ --max_batch_size 1
