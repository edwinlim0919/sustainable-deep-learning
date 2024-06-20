# TensorRT-LLM standalone

sed -i s/016a887123e6/016a887123e6/g cmd_paste_Llama70B_a10040gb.sh

# Flushing GPU memory
sudo fuser -v /dev/nvidia*
sudo kill -9 <PID>

# /dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm
# /TensorRT-LLM/examples/llama
huggingface-cli login
sudo docker cp download_hf_weights.py 016a887123e6:/TensorRT-LLM/examples/llama/download_hf_weights.py
python3 download_hf_weights.py --model-name meta-llama/Llama-2-70b-chat-hf


sudo docker cp meta-llama/ 92a3527e8c38:/app/tensorrt_llm/examples/llama
