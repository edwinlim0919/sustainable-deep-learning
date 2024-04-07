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

# Copy Llama2 tokenizer to inference server model repo
mkdir tensorrt_llm/examples/llama/llama2/
cp -r /dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/meta-llama/Llama-2-7b-chat-hf_tokenizer/* tensorrt_llm/examples/llama/llama2/



# TensorRT-LLM standalone
# /dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/TensorRT-LLM

sudo nvidia-persistenced --user root
sudo make -C docker release_run # starts the NVIDIA docker container

# TODO: May be necessary to run this command sequence if version mismatch error occurs
sudo apt-get --purge remove "*nvidia*"
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
# TODO: Command sequence end

# /dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm
sudo docker ps # tells you NVIDIA docker container id
sed -i "s/5742c720375f/a5ccc32211aa/g" cmd_paste.sh # replace docker container id in cmd_paste.sh with the current one
python3 download_hf_weights.py --model-name "meta-llama/Llama-2-7b-chat-hf"
sudo docker cp meta-llama/ a5ccc32211aa:/app/tensorrt_llm/examples/llama

# /app/tensorrt_llm/examples/llama
pip install nltk
pip install rouge_score
python convert_checkpoint.py --model_dir meta-llama/Llama-2-7b-chat-hf_model --dtype float16 --output_dir ./llama/7B/trt_ckpt/fp16/1-gpu/
trtllm-build --checkpoint_dir ./llama/7B/trt_ckpt/fp16/1-gpu/ --gemm_plugin float16 --output_dir ./llama/7B/trt_engines/fp16/1-gpu-1-batch/ --max_batch_size 1
trtllm-build --checkpoint_dir ./llama/7B/trt_ckpt/fp16/1-gpu/ --gemm_plugin float16 --output_dir ./llama/7B/trt_engines/fp16/1-gpu-2-batch/ --max_batch_size 2
trtllm-build --checkpoint_dir ./llama/7B/trt_ckpt/fp16/1-gpu/ --gemm_plugin float16 --output_dir ./llama/7B/trt_engines/fp16/1-gpu-4-batch/ --max_batch_size 4
trtllm-build --checkpoint_dir ./llama/7B/trt_ckpt/fp16/1-gpu/ --gemm_plugin float16 --output_dir ./llama/7B/trt_engines/fp16/1-gpu-8-batch/ --max_batch_size 8
trtllm-build --checkpoint_dir ./llama/7B/trt_ckpt/fp16/1-gpu/ --gemm_plugin float16 --output_dir ./llama/7B/trt_engines/fp16/1-gpu-16-batch/ --max_batch_size 16
trtllm-build --checkpoint_dir ./llama/7B/trt_ckpt/fp16/1-gpu/ --gemm_plugin float16 --output_dir ./llama/7B/trt_engines/fp16/1-gpu-32-batch/ --max_batch_size 32
trtllm-build --checkpoint_dir ./llama/7B/trt_ckpt/fp16/1-gpu/ --gemm_plugin float16 --output_dir ./llama/7B/trt_engines/fp16/1-gpu-64-batch/ --max_batch_size 64

python ../summarize.py --test_trt_llm --hf_model_dir ./meta-llama/Llama-2-7b-chat-hf_tokenizer --data_type fp16 --engine_dir ./llama/7B/trt_engines/fp16/1-gpu/

# /dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm
sudo docker cp benchmarking/benchmark_trtllm.py a5ccc32211aa:/app/tensorrt_llm/examples/benchmark_trtllm.py
sudo docker cp benchmarking/benchmark_utils.py a5ccc32211aa:/app/tensorrt_llm/examples/benchmark_utils.py
sudo docker cp TensorRT-LLM/examples/summarize.py a5ccc32211aa:/app/tensorrt_llm/examples/summarize.py

sudo docker cp ShareGPT_V3_unfiltered_cleaned_split.json a5ccc32211aa:/app/tensorrt_llm/examples/ShareGPT_V3_unfiltered_cleaned_split.json
sudo docker cp ShareGPT_V3_unfiltered_cleaned_split_top100.json a5ccc32211aa:/app/tensorrt_llm/examples/ShareGPT_V3_unfiltered_cleaned_split_top100.json

# /app/tensorrt_llm/examples/llama
python ../benchmark_trtllm.py --tokenizer_dir ./meta-llama/Llama-2-7b-chat-hf_tokenizer/ --engine_dir ./llama/7B/trt_engines/fp16/1-gpu-1-batch/ --dataset_path ../ShareGPT_V3_unfiltered_cleaned_split_top100.json --num_requests_sample 1 --max_batch_size 1 --max_input_tokens 1000 --max_output_tokens 1000 --output_dir ./outputs/llama/7B/fp16/1-gpu-1-batch --output_file dev_testing.out --random_seed 42
python ../benchmark_trtllm.py --tokenizer_dir ./meta-llama/Llama-2-7b-chat-hf_tokenizer/ --engine_dir ./llama/7B/trt_engines/fp16/1-gpu-1-batch/ --dataset_path ../ShareGPT_V3_unfiltered_cleaned_split_top100.json --num_requests_sample 4 --max_batch_size 1 --max_input_tokens 1000 --max_output_tokens 1000 --output_dir ./outputs/llama/7B/fp16/1-gpu-1-batch --output_file dev_testing.out --random_seed 42


python ../benchmark_trtllm.py --tokenizer_dir ./meta-llama/Llama-2-7b-chat-hf_tokenizer/ --engine_dir ./llama/7B/trt_engines/fp16/1-gpu-2-batch/ --dataset_path ../ShareGPT_V3_unfiltered_cleaned_split_top100.json --num_requests_sample 1 --max_batch_size 2 --max_input_tokens 1000 --max_output_tokens 1000 --output_dir ./outputs/llama/7B/fp16/1-gpu-2-batch --output_file dev_testing.out --random_seed 42
