# TensorRT-LLM standalone

#sed -i "s/5742c720375f/92a3527e8c38/g" cmd_paste_Llama13B.sh # replace docker container id in cmd_paste.sh with the current one
sed -i "s/92a3527e8c38/92a3527e8c38/g" cmd_paste_Llama13B_a10040gb.sh

# Flushing GPU memory
sudo fuser -v /dev/nvidia*
sudo kill -9 <PID>

# /dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm
huggingface-cli login
python3 download_hf_weights.py --model-name "meta-llama/Llama-2-13b-chat-hf"
sudo docker cp meta-llama/ 92a3527e8c38:/app/tensorrt_llm/examples/llama

# /app/tensorrt_llm/examples/llama
pip install nltk
pip install rouge_score
pip install aiofiles
python convert_checkpoint.py --model_dir meta-llama/Llama-2-13b-chat-hf_model --dtype float16 --output_dir ./llama/13B/trt_ckpt/fp16/1-gpu/ --load_by_shard
trtllm-build --checkpoint_dir ./llama/13B/trt_ckpt/fp16/1-gpu/ --gemm_plugin float16 --output_dir ./llama/13B/trt_engines/fp16/1-gpu-1-batch/ --max_batch_size 1
trtllm-build --checkpoint_dir ./llama/13B/trt_ckpt/fp16/1-gpu/ --gemm_plugin float16 --output_dir ./llama/13B/trt_engines/fp16/1-gpu-2-batch/ --max_batch_size 2
trtllm-build --checkpoint_dir ./llama/13B/trt_ckpt/fp16/1-gpu/ --gemm_plugin float16 --output_dir ./llama/13B/trt_engines/fp16/1-gpu-4-batch/ --max_batch_size 4
trtllm-build --checkpoint_dir ./llama/13B/trt_ckpt/fp16/1-gpu/ --gemm_plugin float16 --output_dir ./llama/13B/trt_engines/fp16/1-gpu-6-batch/ --max_batch_size 6
trtllm-build --checkpoint_dir ./llama/13B/trt_ckpt/fp16/1-gpu/ --gemm_plugin float16 --output_dir ./llama/13B/trt_engines/fp16/1-gpu-8-batch/ --max_batch_size 8

# /dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm
sudo docker cp benchmarking/benchmark_trtllm.py 92a3527e8c38:/app/tensorrt_llm/examples/benchmark_trtllm.py
sudo docker cp benchmarking/benchmark_utils.py 92a3527e8c38:/app/tensorrt_llm/examples/benchmark_utils.py
sudo docker cp ShareGPT_V3_unfiltered_cleaned_split.json 92a3527e8c38:/app/tensorrt_llm/examples/ShareGPT_V3_unfiltered_cleaned_split.json
sudo docker cp ShareGPT_V3_unfiltered_cleaned_split_top100.json 92a3527e8c38:/app/tensorrt_llm/examples/ShareGPT_V3_unfiltered_cleaned_split_top100.json

# experiments
# 1 gpu 1 batch 1000 max
# /dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm
python benchmarking/nvsmi_monitor.py --output_dir outputs/llama/13B/fp16/1-gpu-1-batch --output_file nvsmi_numreqsample0_iter100_max1000_a10040gb.out --container_id 92a3527e8c38 --container_output_dir /app/tensorrt_llm/examples/llama --container_stop_file container_stop.txt --gpu_type a10040gb
# /app/tensorrt_llm/examples/llama
python ../benchmark_trtllm.py --tokenizer_dir ./meta-llama/Llama-2-13b-chat-hf_tokenizer/ --engine_dir ./llama/13B/trt_engines/fp16/1-gpu-1-batch/ --dataset_path ../ShareGPT_V3_unfiltered_cleaned_split.json --num_requests_sample 0 --max_batch_size 1 --max_input_tokens 1000 --max_output_tokens 1000 --output_dir /app/tensorrt_llm/examples/llama/outputs/llama/13B/fp16/1-gpu-1-batch --output_file bmark_numreqsample0_iter100_max1000_a10040gb.out --container_output_dir /app/tensorrt_llm/examples/llama --container_stop_file container_stop.txt --random_seed 42 --num_iterations 100 --use_prompt_formatting --add_special_tokens
# /dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm
sudo docker cp 92a3527e8c38:/app/tensorrt_llm/examples/llama/outputs/llama/13B/fp16/1-gpu-1-batch/bmark_numreqsample0_iter100_max1000_a10040gb.out outputs/llama/13B/fp16/1-gpu-1-batch/bmark_numreqsample0_iter100_max1000_a10040gb.out

# 1 gpu 2 batch 1000 max
# /dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm
python benchmarking/nvsmi_monitor.py --output_dir outputs/llama/13B/fp16/1-gpu-2-batch --output_file nvsmi_numreqsample0_iter100_max1000_a10040gb.out --container_id 92a3527e8c38 --container_output_dir /app/tensorrt_llm/examples/llama --container_stop_file container_stop.txt --gpu_type a10040gb
# /app/tensorrt_llm/examples/llama
python ../benchmark_trtllm.py --tokenizer_dir ./meta-llama/Llama-2-13b-chat-hf_tokenizer/ --engine_dir ./llama/13B/trt_engines/fp16/1-gpu-2-batch/ --dataset_path ../ShareGPT_V3_unfiltered_cleaned_split.json --num_requests_sample 0 --max_batch_size 2 --max_input_tokens 1000 --max_output_tokens 1000 --output_dir /app/tensorrt_llm/examples/llama/outputs/llama/13B/fp16/1-gpu-2-batch --output_file bmark_numreqsample0_iter100_max1000_a10040gb.out --container_output_dir /app/tensorrt_llm/examples/llama --container_stop_file container_stop.txt --random_seed 42 --num_iterations 100 --use_prompt_formatting --add_special_tokens
# /dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm
sudo docker cp 92a3527e8c38:/app/tensorrt_llm/examples/llama/outputs/llama/13B/fp16/1-gpu-2-batch/bmark_numreqsample0_iter100_max1000_a10040gb.out outputs/llama/13B/fp16/1-gpu-2-batch/bmark_numreqsample0_iter100_max1000_a10040gb.out
