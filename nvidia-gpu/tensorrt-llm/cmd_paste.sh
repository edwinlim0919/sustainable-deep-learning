## Llama2 7B fp16 1GPU
# Downloading weights
# /dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm
conda activate tensorrtllm_backend
huggingface-cli login
python3 download_hf_weights.py --model-name "meta-llama/Llama-2-7b-chat-hf"

# Creating TRT engine and testing
# /dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/tensorrtllm_backend/tensorrt_llm/examples/llama
pip install -r requirements.txt
python3 convert_checkpoint.py --model_dir /dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/meta-llama/Llama-2-7b-chat-hf_model --output_dir ./tllm_checkpoint_1gpu_fp16 --dtype float16
trtllm-build --checkpoint_dir ./tllm_checkpoint_1gpu_fp16 --output_dir ./tmp/llama/7B/trt_engines/fp16/1-gpu --gemm_plugin float16
python3 ../run.py --max_output_len=50 --tokenizer_dir /dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/meta-llama/Llama-2-7b-chat-hf_tokenizer --engine_dir=./tmp/llama/7B/trt_engines/fp16/1-gpu/
python3 ../summarize.py --test_trt_llm --hf_model_dir /dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/meta-llama/Llama-2-7b-chat-hf_tokenizer --data_type fp16 --engine_dir=./tmp/llama/7B/trt_engines/fp16/1-gpu/

# Copy TRT engine to triton inference server model repo
# /dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/tensorrtllm_backend
cp -r all_models/inflight_batcher_llm/* triton_model_repo/
cp tensorrt_llm/examples/llama/tmp/llama/7B/trt_engines/fp16/1-gpu/* triton_model_repo/tensorrt_llm/1

# Modify the model configuration
# /dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/tensorrtllm_backend
sed -i 's|\${tokenizer_dir}|/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/meta-llama/Llama-2-7b-chat-hf_tokenizer|g' triton_model_repo/preprocessing/config.pbtxt
sed -i 's|\${batching_strategy}|inflight_fused_batching|g' triton_model_repo/tensorrt_llm/config.pbtxt
sed -i 's|\${engine_dir}|/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/tensorrtllm_backend/triton_model_repo/tensorrt_llm/1|g' triton_model_repo/tensorrt_llm/config.pbtxt
sed -i 's|\${batch_scheduler_policy}|max_utilization|g' triton_model_repo/tensorrt_llm/config.pbtxt
sed -i 's|\${gpu_device_ids}|0|g' triton_model_repo/tensorrt_llm/config.pbtxt
sed -i 's|\${tokenizer_dir}|/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/meta-llama/Llama-2-7b-chat-hf_tokenizer|g' triton_model_repo/postprocessing/config.pbtxt

# Launch Triton server
# /dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/tensorrtllm_backend
sudo docker run --rm -it --net host --shm-size=2g --ulimit memlock=-1 --ulimit stack=67108864 --gpus all -v /dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/tensorrtllm_backend:/tensorrtllm_backend tritonserver bash
