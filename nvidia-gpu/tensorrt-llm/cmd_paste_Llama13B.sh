# TensorRT-LLM standalone

sed -i "s/5742c720375f/f88488e6d7ff/g" cmd_paste_Llama13B.sh # replace docker container id in cmd_paste.sh with the current one

# /dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm
huggingface-cli login
python3 download_hf_weights.py --model-name "meta-llama/Llama-2-13b-chat-hf"
sudo docker cp meta-llama/ f88488e6d7ff:/app/tensorrt_llm/examples/llama

# /app/tensorrt_llm/examples/llama
pip install nltk
pip install rouge_score
pip install aiofiles
python convert_checkpoint.py --model_dir meta-llama/Llama-2-13b-chat-hf_model --dtype float16 --output_dir ./llama/13B/trt_ckpt/fp16/1-gpu/ --load_by_shard
trtllm-build --checkpoint_dir ./llama/13B/trt_ckpt/fp16/1-gpu/ --gemm_plugin float16 --output_dir ./llama/13B/trt_engines/fp16/1-gpu-1-batch/ --max_batch_size 1

# /dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm
sudo docker cp benchmarking/benchmark_trtllm.py f88488e6d7ff:/app/tensorrt_llm/examples/benchmark_trtllm.py
sudo docker cp benchmarking/benchmark_utils.py f88488e6d7ff:/app/tensorrt_llm/examples/benchmark_utils.py
sudo docker cp ShareGPT_V3_unfiltered_cleaned_split_top100.json f88488e6d7ff:/app/tensorrt_llm/examples/ShareGPT_V3_unfiltered_cleaned_split_top100.json

# dev testing
# /app/tensorrt_llm/examples/llama
python ../benchmark_trtllm.py --tokenizer_dir ./meta-llama/Llama-2-13b-chat-hf_tokenizer/ --engine_dir ./llama/13B/trt_engines/fp16/1-gpu-1-batch/ --dataset_path ../ShareGPT_V3_unfiltered_cleaned_split_top100.json --num_requests_sample 4 --max_batch_size 1 --max_input_tokens 1000 --max_output_tokens 1000 --output_dir ./outputs/llama/13B/fp16/1-gpu-1-batch --output_file dev_testing.out --random_seed 42 --num_iterations 10
# /dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm
sudo docker cp f88488e6d7ff:/app/tensorrt_llm/examples/llama/outputs/llama/13B/fp16/1-gpu-1-batch/dev_testing.out dev_testing.out

# experiments
# 1 gpu 1 batch
# /dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm
python benchmarking/nvsmi_monitor.py --output_dir outputs/llama/7B/fp16/1-gpu-1-batch --output_file nvsmi_numreqsample10000_iter100.out
# /app/tensorrt_llm/examples/llama
python ../benchmark_trtllm.py --tokenizer_dir ./meta-llama/Llama-2-7b-chat-hf_tokenizer/ --engine_dir ./llama/7B/trt_engines/fp16/1-gpu-1-batch/ --dataset_path ../ShareGPT_V3_unfiltered_cleaned_split.json --num_requests_sample 10000 --max_batch_size 1 --max_input_tokens 1000 --max_output_tokens 1000 --output_dir ./outputs/llama/7B/fp16/1-gpu-1-batch --output_file bmark_numreqsample10000_iter100.out --random_seed 42 --num_iterations 100
# /dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm
sudo docker cp f88488e6d7ff:/app/tensorrt_llm/examples/llama/outputs/llama/7B/fp16/1-gpu-1-batch/bmark_numreqsample10000_iter100.out outputs/llama/7B/fp16/1-gpu-1-batch/bmark_numreqsample10000_iter100.out
