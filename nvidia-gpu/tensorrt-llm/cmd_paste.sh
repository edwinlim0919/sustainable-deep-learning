# Llama2 7B fp16 1GPU
conda activate tensorrt-llm
huggingface-cli login
python3 download_hf_weights.py --model-name "meta-llama/Llama-2-7b-chat-hf"

cd TensorRT-LLM/examples/llama
pip install -r requirements.txt
python3 convert_checkpoint.py --model_dir /dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/meta-llama/Llama-2-7b-chat-hf_model --output_dir ./tllm_checkpoint_1gpu_fp16 --dtype float16
trtllm-build --checkpoint_dir ./tllm_checkpoint_1gpu_fp16 --output_dir ./tmp/llama/7B/trt_engines/fp16/1-gpu --gemm_plugin float16

python3 ../run.py --max_output_len=50 --tokenizer_dir /dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/meta-llama/Llama-2-7b-chat-hf_tokenizer --engine_dir=./tmp/llama/7B/trt_engines/fp16/1-gpu/
python3 ../summarize.py --test_trt_llm --hf_model_dir /dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/meta-llama/Llama-2-7b-chat-hf_tokenizer --data_type fp16 --engine_dir=./tmp/llama/7B/trt_engines/fp16/1-gpu/
