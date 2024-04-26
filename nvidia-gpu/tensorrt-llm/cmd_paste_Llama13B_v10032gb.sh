# TensorRT-LLM standalone

sed -i "s/f88488e6d7ff/c71f5b2d5461/g" cmd_paste_Llama13B_v10032gb.sh # replace docker container id in cmd_paste.sh with the current one

# Flushing GPU memory
sudo fuser -v /dev/nvidia*
sudo kill -9 <PID>

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
trtllm-build --checkpoint_dir ./llama/13B/trt_ckpt/fp16/1-gpu/ --gemm_plugin float16 --output_dir ./llama/13B/trt_engines/fp16/1-gpu-2-batch/ --max_batch_size 2
trtllm-build --checkpoint_dir ./llama/13B/trt_ckpt/fp16/1-gpu/ --gemm_plugin float16 --output_dir ./llama/13B/trt_engines/fp16/1-gpu-4-batch/ --max_batch_size 4
trtllm-build --checkpoint_dir ./llama/13B/trt_ckpt/fp16/1-gpu/ --gemm_plugin float16 --output_dir ./llama/13B/trt_engines/fp16/1-gpu-6-batch/ --max_batch_size 6
trtllm-build --checkpoint_dir ./llama/13B/trt_ckpt/fp16/1-gpu/ --gemm_plugin float16 --output_dir ./llama/13B/trt_engines/fp16/1-gpu-8-batch/ --max_batch_size 8

# 4-BIT WEIGHT QUANTIZED
python convert_checkpoint.py --model_dir meta-llama/Llama-2-13b-chat-hf_model --output_dir ./llama/13B/trt_ckpt/fp16_wq4/1-gpu/ --dtype float16 --use_weight_only --weight_only_precision int4
trtllm-build --checkpoint_dir ./llama/13B/trt_ckpt/fp16_wq4/1-gpu/ --gemm_plugin float16 --output_dir ./llama/13B/trt_engines/fp16_wq4/1-gpu-1-batch/ --max_batch_size 1
trtllm-build --checkpoint_dir ./llama/13B/trt_ckpt/fp16_wq4/1-gpu/ --gemm_plugin float16 --output_dir ./llama/13B/trt_engines/fp16_wq4/1-gpu-4-batch/ --max_batch_size 4
trtllm-build --checkpoint_dir ./llama/13B/trt_ckpt/fp16_wq4/1-gpu/ --gemm_plugin float16 --output_dir ./llama/13B/trt_engines/fp16_wq4/1-gpu-6-batch/ --max_batch_size 6
trtllm-build --checkpoint_dir ./llama/13B/trt_ckpt/fp16_wq4/1-gpu/ --gemm_plugin float16 --output_dir ./llama/13B/trt_engines/fp16_wq4/1-gpu-8-batch/ --max_batch_size 8
trtllm-build --checkpoint_dir ./llama/13B/trt_ckpt/fp16_wq4/1-gpu/ --gemm_plugin float16 --output_dir ./llama/13B/trt_engines/fp16_wq4/1-gpu-10-batch/ --max_batch_size 10

# /dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm
sudo docker cp benchmarking/benchmark_trtllm.py f88488e6d7ff:/app/tensorrt_llm/examples/benchmark_trtllm.py
sudo docker cp benchmarking/benchmark_utils.py f88488e6d7ff:/app/tensorrt_llm/examples/benchmark_utils.py
sudo docker cp ShareGPT_V3_unfiltered_cleaned_split.json f88488e6d7ff:/app/tensorrt_llm/examples/ShareGPT_V3_unfiltered_cleaned_split.json
sudo docker cp ShareGPT_V3_unfiltered_cleaned_split_top100.json f88488e6d7ff:/app/tensorrt_llm/examples/ShareGPT_V3_unfiltered_cleaned_split_top100.json


# experiments
# 1 gpu 1 batch 1000 max
# /dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm
python benchmarking/nvsmi_monitor.py --output_dir outputs/llama/13B/fp16/1-gpu-1-batch --output_file nvsmi_numreqsample0_iter100_max1000.out
# /app/tensorrt_llm/examples/llama
python ../benchmark_trtllm.py --tokenizer_dir ./meta-llama/Llama-2-13b-chat-hf_tokenizer/ --engine_dir ./llama/13B/trt_engines/fp16/1-gpu-1-batch/ --dataset_path ../ShareGPT_V3_unfiltered_cleaned_split.json --num_requests_sample 0 --max_batch_size 1 --max_input_tokens 1000 --max_output_tokens 1000 --output_dir ./outputs/llama/13B/fp16/1-gpu-1-batch --output_file bmark_numreqsample0_iter100_max1000.out --random_seed 42 --num_iterations 100 --use_prompt_formatting --add_special_tokens
# /dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm
sudo docker cp f88488e6d7ff:/app/tensorrt_llm/examples/llama/outputs/llama/13B/fp16/1-gpu-1-batch/bmark_numreqsample0_iter100_max1000.out outputs/llama/13B/fp16/1-gpu-1-batch/bmark_numreqsample0_iter100_max1000.out

# 1 gpu 1 batch 500 max
# /dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm
python benchmarking/nvsmi_monitor.py --output_dir outputs/llama/13B/fp16/1-gpu-1-batch --output_file nvsmi_numreqsample0_iter100_max500.out
# /app/tensorrt_llm/examples/llama
python ../benchmark_trtllm.py --tokenizer_dir ./meta-llama/Llama-2-13b-chat-hf_tokenizer/ --engine_dir ./llama/13B/trt_engines/fp16/1-gpu-1-batch/ --dataset_path ../ShareGPT_V3_unfiltered_cleaned_split.json --num_requests_sample 0 --max_batch_size 1 --max_input_tokens 500 --max_output_tokens 500 --output_dir ./outputs/llama/13B/fp16/1-gpu-1-batch --output_file bmark_numreqsample0_iter100_max500.out --random_seed 42 --num_iterations 100 --use_prompt_formatting --add_special_tokens
# /dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm
sudo docker cp f88488e6d7ff:/app/tensorrt_llm/examples/llama/outputs/llama/13B/fp16/1-gpu-1-batch/bmark_numreqsample0_iter100_max500.out outputs/llama/13B/fp16/1-gpu-1-batch/bmark_numreqsample0_iter100_max500.out

# 1 gpu 2 batch 1000 max
# /dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm
python benchmarking/nvsmi_monitor.py --output_dir outputs/llama/13B/fp16/1-gpu-2-batch --output_file nvsmi_numreqsample0_iter100_max1000.out
# /app/tensorrt_llm/examples/llama
python ../benchmark_trtllm.py --tokenizer_dir ./meta-llama/Llama-2-13b-chat-hf_tokenizer/ --engine_dir ./llama/13B/trt_engines/fp16/1-gpu-2-batch/ --dataset_path ../ShareGPT_V3_unfiltered_cleaned_split.json --num_requests_sample 0 --max_batch_size 2 --max_input_tokens 1000 --max_output_tokens 1000 --output_dir ./outputs/llama/13B/fp16/1-gpu-2-batch --output_file bmark_numreqsample0_iter100_max1000.out --random_seed 42 --num_iterations 100 --use_prompt_formatting --add_special_tokens
# /dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm
sudo docker cp f88488e6d7ff:/app/tensorrt_llm/examples/llama/outputs/llama/13B/fp16/1-gpu-2-batch/bmark_numreqsample0_iter100_max1000.out outputs/llama/13B/fp16/1-gpu-2-batch/bmark_numreqsample0_iter100_max1000.out

# 1 gpu 2 batch 500 max
# /dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm
python benchmarking/nvsmi_monitor.py --output_dir outputs/llama/13B/fp16/1-gpu-2-batch --output_file nvsmi_numreqsample0_iter100_max500.out
# /app/tensorrt_llm/examples/llama
python ../benchmark_trtllm.py --tokenizer_dir ./meta-llama/Llama-2-13b-chat-hf_tokenizer/ --engine_dir ./llama/13B/trt_engines/fp16/1-gpu-2-batch/ --dataset_path ../ShareGPT_V3_unfiltered_cleaned_split.json --num_requests_sample 0 --max_batch_size 2 --max_input_tokens 500 --max_output_tokens 500 --output_dir ./outputs/llama/13B/fp16/1-gpu-2-batch --output_file bmark_numreqsample0_iter100_max500.out --random_seed 42 --num_iterations 100 --use_prompt_formatting --add_special_tokens
# /dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm
sudo docker cp f88488e6d7ff:/app/tensorrt_llm/examples/llama/outputs/llama/13B/fp16/1-gpu-2-batch/bmark_numreqsample0_iter100_max500.out outputs/llama/13B/fp16/1-gpu-2-batch/bmark_numreqsample0_iter100_max500.out

# 1 gpu 4 batch 1000 max
# /dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm
python benchmarking/nvsmi_monitor.py --output_dir outputs/llama/13B/fp16/1-gpu-4-batch --output_file nvsmi_numreqsample0_iter100_max1000.out
# /app/tensorrt_llm/examples/llama
python ../benchmark_trtllm.py --tokenizer_dir ./meta-llama/Llama-2-13b-chat-hf_tokenizer/ --engine_dir ./llama/13B/trt_engines/fp16/1-gpu-4-batch/ --dataset_path ../ShareGPT_V3_unfiltered_cleaned_split.json --num_requests_sample 0 --max_batch_size 4 --max_input_tokens 1000 --max_output_tokens 1000 --output_dir ./outputs/llama/13B/fp16/1-gpu-4-batch --output_file bmark_numreqsample0_iter100_max1000.out --random_seed 42 --num_iterations 100 --use_prompt_formatting --add_special_tokens
# /dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm
sudo docker cp f88488e6d7ff:/app/tensorrt_llm/examples/llama/outputs/llama/13B/fp16/1-gpu-4-batch/bmark_numreqsample0_iter100_max1000.out outputs/llama/13B/fp16/1-gpu-4-batch/bmark_numreqsample0_iter100_max1000.out

# 1 gpu 4 batch 500 max
# /dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm
python benchmarking/nvsmi_monitor.py --output_dir outputs/llama/13B/fp16/1-gpu-4-batch --output_file nvsmi_numreqsample0_iter100_max500.out
# /app/tensorrt_llm/examples/llama
python ../benchmark_trtllm.py --tokenizer_dir ./meta-llama/Llama-2-13b-chat-hf_tokenizer/ --engine_dir ./llama/13B/trt_engines/fp16/1-gpu-4-batch/ --dataset_path ../ShareGPT_V3_unfiltered_cleaned_split.json --num_requests_sample 0 --max_batch_size 4 --max_input_tokens 500 --max_output_tokens 500 --output_dir ./outputs/llama/13B/fp16/1-gpu-4-batch --output_file bmark_numreqsample0_iter100_max500.out --random_seed 42 --num_iterations 100 --use_prompt_formatting --add_special_tokens
# /dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm
sudo docker cp f88488e6d7ff:/app/tensorrt_llm/examples/llama/outputs/llama/13B/fp16/1-gpu-4-batch/bmark_numreqsample0_iter100_max500.out outputs/llama/13B/fp16/1-gpu-4-batch/bmark_numreqsample0_iter100_max500.out

# TODO: OOM
# 1 gpu 6 batch 1000 max
# /dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm
python benchmarking/nvsmi_monitor.py --output_dir outputs/llama/13B/fp16/1-gpu-6-batch --output_file nvsmi_numreqsample0_iter100_max1000.out
# /app/tensorrt_llm/examples/llama
python ../benchmark_trtllm.py --tokenizer_dir ./meta-llama/Llama-2-13b-chat-hf_tokenizer/ --engine_dir ./llama/13B/trt_engines/fp16/1-gpu-6-batch/ --dataset_path ../ShareGPT_V3_unfiltered_cleaned_split.json --num_requests_sample 0 --max_batch_size 6 --max_input_tokens 1000 --max_output_tokens 1000 --output_dir ./outputs/llama/13B/fp16/1-gpu-6-batch --output_file bmark_numreqsample0_iter100_max1000.out --random_seed 42 --num_iterations 100 --use_prompt_formatting --add_special_tokens
# /dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm
sudo docker cp f88488e6d7ff:/app/tensorrt_llm/examples/llama/outputs/llama/13B/fp16/1-gpu-6-batch/bmark_numreqsample0_iter100_max1000.out outputs/llama/13B/fp16/1-gpu-6-batch/bmark_numreqsample0_iter100_max1000.out

# 1 gpu 6 batch 500 max
# /dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm
python benchmarking/nvsmi_monitor.py --output_dir outputs/llama/13B/fp16/1-gpu-6-batch --output_file nvsmi_numreqsample0_iter100_max500.out
# /app/tensorrt_llm/examples/llama
python ../benchmark_trtllm.py --tokenizer_dir ./meta-llama/Llama-2-13b-chat-hf_tokenizer/ --engine_dir ./llama/13B/trt_engines/fp16/1-gpu-6-batch/ --dataset_path ../ShareGPT_V3_unfiltered_cleaned_split.json --num_requests_sample 0 --max_batch_size 6 --max_input_tokens 500 --max_output_tokens 500 --output_dir ./outputs/llama/13B/fp16/1-gpu-6-batch --output_file bmark_numreqsample0_iter100_max500.out --random_seed 42 --num_iterations 100 --use_prompt_formatting --add_special_tokens
# /dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm
sudo docker cp f88488e6d7ff:/app/tensorrt_llm/examples/llama/outputs/llama/13B/fp16/1-gpu-6-batch/bmark_numreqsample0_iter100_max500.out outputs/llama/13B/fp16/1-gpu-6-batch/bmark_numreqsample0_iter100_max500.out

# TODO: OOM
# 1 gpu 8 batch 500 max
# /dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm
python benchmarking/nvsmi_monitor.py --output_dir outputs/llama/13B/fp16/1-gpu-8-batch --output_file nvsmi_numreqsample0_iter100_max500.out
# /app/tensorrt_llm/examples/llama
python ../benchmark_trtllm.py --tokenizer_dir ./meta-llama/Llama-2-13b-chat-hf_tokenizer/ --engine_dir ./llama/13B/trt_engines/fp16/1-gpu-8-batch/ --dataset_path ../ShareGPT_V3_unfiltered_cleaned_split.json --num_requests_sample 0 --max_batch_size 8 --max_input_tokens 500 --max_output_tokens 500 --output_dir ./outputs/llama/13B/fp16/1-gpu-8-batch --output_file bmark_numreqsample0_iter100_max500.out --random_seed 42 --num_iterations 100 --use_prompt_formatting --add_special_tokens
# /dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm
sudo docker cp f88488e6d7ff:/app/tensorrt_llm/examples/llama/outputs/llama/13B/fp16/1-gpu-8-batch/bmark_numreqsample0_iter100_max500.out outputs/llama/13B/fp16/1-gpu-8-batch/bmark_numreqsample0_iter100_max500.out




# 4-BIT WEIGHT-QUANTIZED
# 1 gpu 1 batch 1000 max
# /dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm
python benchmarking/nvsmi_monitor.py --output_dir outputs/llama/13B/fp16_wq4/1-gpu-1-batch --output_file nvsmi_numreqsample0_iter100_max1000_v10032gb.out --container_id c71f5b2d5461 --container_output_dir /app/tensorrt_llm/examples/llama --container_stop_file container_stop.txt --gpu_type v10032gb
# /app/tensorrt_llm/examples/llama
python ../benchmark_trtllm.py --tokenizer_dir ./meta-llama/Llama-2-13b-chat-hf_tokenizer/ --engine_dir ./llama/13B/trt_engines/fp16_wq4/1-gpu-1-batch/ --dataset_path ../ShareGPT_V3_unfiltered_cleaned_split.json --num_requests_sample 0 --max_batch_size 1 --max_input_tokens 1000 --max_output_tokens 1000 --output_dir /app/tensorrt_llm/examples/llama/outputs/llama/13B/fp16_wq4/1-gpu-1-batch --output_file bmark_numreqsample0_iter100_max1000_v10032gb.out --container_output_dir /app/tensorrt_llm/examples/llama --container_stop_file container_stop.txt --random_seed 42 --num_iterations 100 --use_prompt_formatting --add_special_tokens
# /dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm
sudo docker cp c71f5b2d5461:/app/tensorrt_llm/examples/llama/outputs/llama/13B/fp16_wq4/1-gpu-1-batch/bmark_numreqsample0_iter100_max1000_v10032gb.out outputs/llama/13B/fp16_wq4/1-gpu-1-batch/bmark_numreqsample0_iter100_max1000_v10032gb.out

# 1 gpu 4 batch 1000 max
# /dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm
python benchmarking/nvsmi_monitor.py --output_dir outputs/llama/13B/fp16_wq4/1-gpu-4-batch --output_file nvsmi_numreqsample0_iter100_max1000_v10032gb.out --container_id c71f5b2d5461 --container_output_dir /app/tensorrt_llm/examples/llama --container_stop_file container_stop.txt --gpu_type v10032gb
# /app/tensorrt_llm/examples/llama
python ../benchmark_trtllm.py --tokenizer_dir ./meta-llama/Llama-2-13b-chat-hf_tokenizer/ --engine_dir ./llama/13B/trt_engines/fp16_wq4/1-gpu-4-batch/ --dataset_path ../ShareGPT_V3_unfiltered_cleaned_split.json --num_requests_sample 0 --max_batch_size 4 --max_input_tokens 1000 --max_output_tokens 1000 --output_dir /app/tensorrt_llm/examples/llama/outputs/llama/13B/fp16_wq4/1-gpu-4-batch --output_file bmark_numreqsample0_iter100_max1000_v10032gb.out --container_output_dir /app/tensorrt_llm/examples/llama --container_stop_file container_stop.txt --random_seed 42 --num_iterations 100 --use_prompt_formatting --add_special_tokens
# /dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm
sudo docker cp c71f5b2d5461:/app/tensorrt_llm/examples/llama/outputs/llama/13B/fp16_wq4/1-gpu-4-batch/bmark_numreqsample0_iter100_max1000_v10032gb.out outputs/llama/13B/fp16_wq4/1-gpu-4-batch/bmark_numreqsample0_iter100_max1000_v10032gb.out

# 1 gpu 6 batch 1000 max
# /dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm
python benchmarking/nvsmi_monitor.py --output_dir outputs/llama/13B/fp16_wq4/1-gpu-6-batch --output_file nvsmi_numreqsample0_iter100_max1000_v10032gb.out --container_id c71f5b2d5461 --container_output_dir /app/tensorrt_llm/examples/llama --container_stop_file container_stop.txt --gpu_type v10032gb
# /app/tensorrt_llm/examples/llama
python ../benchmark_trtllm.py --tokenizer_dir ./meta-llama/Llama-2-13b-chat-hf_tokenizer/ --engine_dir ./llama/13B/trt_engines/fp16_wq4/1-gpu-6-batch/ --dataset_path ../ShareGPT_V3_unfiltered_cleaned_split.json --num_requests_sample 0 --max_batch_size 6 --max_input_tokens 1000 --max_output_tokens 1000 --output_dir /app/tensorrt_llm/examples/llama/outputs/llama/13B/fp16_wq4/1-gpu-6-batch --output_file bmark_numreqsample0_iter100_max1000_v10032gb.out --container_output_dir /app/tensorrt_llm/examples/llama --container_stop_file container_stop.txt --random_seed 42 --num_iterations 100 --use_prompt_formatting --add_special_tokens
# /dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm
sudo docker cp c71f5b2d5461:/app/tensorrt_llm/examples/llama/outputs/llama/13B/fp16_wq4/1-gpu-6-batch/bmark_numreqsample0_iter100_max1000_v10032gb.out outputs/llama/13B/fp16_wq4/1-gpu-6-batch/bmark_numreqsample0_iter100_max1000_v10032gb.out

# 1 gpu 8 batch 1000 max
# /dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm
python benchmarking/nvsmi_monitor.py --output_dir outputs/llama/13B/fp16_wq4/1-gpu-8-batch --output_file nvsmi_numreqsample0_iter100_max1000_v10032gb.out --container_id c71f5b2d5461 --container_output_dir /app/tensorrt_llm/examples/llama --container_stop_file container_stop.txt --gpu_type v10032gb
# /app/tensorrt_llm/examples/llama
python ../benchmark_trtllm.py --tokenizer_dir ./meta-llama/Llama-2-13b-chat-hf_tokenizer/ --engine_dir ./llama/13B/trt_engines/fp16_wq4/1-gpu-8-batch/ --dataset_path ../ShareGPT_V3_unfiltered_cleaned_split.json --num_requests_sample 0 --max_batch_size 8 --max_input_tokens 1000 --max_output_tokens 1000 --output_dir /app/tensorrt_llm/examples/llama/outputs/llama/13B/fp16_wq4/1-gpu-8-batch --output_file bmark_numreqsample0_iter100_max1000_v10032gb.out --container_output_dir /app/tensorrt_llm/examples/llama --container_stop_file container_stop.txt --random_seed 42 --num_iterations 100 --use_prompt_formatting --add_special_tokens
# /dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm
sudo docker cp c71f5b2d5461:/app/tensorrt_llm/examples/llama/outputs/llama/13B/fp16_wq4/1-gpu-8-batch/bmark_numreqsample0_iter100_max1000_v10032gb.out outputs/llama/13B/fp16_wq4/1-gpu-8-batch/bmark_numreqsample0_iter100_max1000_v10032gb.out

# 1 gpu 10 batch 1000 max
# /dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm
python benchmarking/nvsmi_monitor.py --output_dir outputs/llama/13B/fp16_wq4/1-gpu-10-batch --output_file nvsmi_numreqsample0_iter100_max1000_v10032gb.out --container_id c71f5b2d5461 --container_output_dir /app/tensorrt_llm/examples/llama --container_stop_file container_stop.txt --gpu_type v10032gb
# /app/tensorrt_llm/examples/llama
python ../benchmark_trtllm.py --tokenizer_dir ./meta-llama/Llama-2-13b-chat-hf_tokenizer/ --engine_dir ./llama/13B/trt_engines/fp16_wq4/1-gpu-10-batch/ --dataset_path ../ShareGPT_V3_unfiltered_cleaned_split.json --num_requests_sample 0 --max_batch_size 10 --max_input_tokens 1000 --max_output_tokens 1000 --output_dir /app/tensorrt_llm/examples/llama/outputs/llama/13B/fp16_wq4/1-gpu-10-batch --output_file bmark_numreqsample0_iter100_max1000_v10032gb.out --container_output_dir /app/tensorrt_llm/examples/llama --container_stop_file container_stop.txt --random_seed 42 --num_iterations 100 --use_prompt_formatting --add_special_tokens
# /dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm
sudo docker cp c71f5b2d5461:/app/tensorrt_llm/examples/llama/outputs/llama/13B/fp16_wq4/1-gpu-10-batch/bmark_numreqsample0_iter100_max1000_v10032gb.out outputs/llama/13B/fp16_wq4/1-gpu-10-batch/bmark_numreqsample0_iter100_max1000_v10032gb.out
