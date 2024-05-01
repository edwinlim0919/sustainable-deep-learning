# TensorRT-LLM standalone

sed -i "s/b209d39f9c48/b209d39f9c48/g" cmd_paste_gpt2_812M_v10032gb.sh # replace docker container id in cmd_paste.sh with the current one

# Flushing GPU memory
sudo fuser -v /dev/nvidia*
sudo kill -9 <PID>

# container "/app/tensorrt_llm/examples/gpt"
pip install -r requirements.txt
rm -rf gpt2-large && git clone https://huggingface.co/gpt2-large gpt2-large
pushd gpt2-large && rm pytorch_model.bin model.safetensors && wget -q https://huggingface.co/gpt2-large/resolve/main/pytorch_model.bin && popd

# 1-gpu fp16 nowq
python3 convert_checkpoint.py --model_dir gpt2-large --dtype float16 --output_dir gpt2-large/trt_ckpt/fp16/1-gpu/
trtllm-build --checkpoint_dir gpt2-large/trt_ckpt/fp16/1-gpu/ --gemm_plugin float16 --output_dir gpt2-large/trt_engines/fp16/1-gpu-1-batch/ --max_batch_size 1
trtllm-build --checkpoint_dir gpt2-large/trt_ckpt/fp16/1-gpu/ --gemm_plugin float16 --output_dir gpt2-large/trt_engines/fp16/1-gpu-128-batch/ --max_batch_size 128
trtllm-build --checkpoint_dir gpt2-large/trt_ckpt/fp16/1-gpu/ --gemm_plugin float16 --output_dir gpt2-large/trt_engines/fp16/1-gpu-136-batch/ --max_batch_size 136

# OOM
trtllm-build --checkpoint_dir gpt2-large/trt_ckpt/fp16/1-gpu/ --gemm_plugin float16 --output_dir gpt2-large/trt_engines/fp16/1-gpu-144-batch/ --max_batch_size 144
trtllm-build --checkpoint_dir gpt2-large/trt_ckpt/fp16/1-gpu/ --gemm_plugin float16 --output_dir gpt2-large/trt_engines/fp16/1-gpu-160-batch/ --max_batch_size 160
trtllm-build --checkpoint_dir gpt2-large/trt_ckpt/fp16/1-gpu/ --gemm_plugin float16 --output_dir gpt2-large/trt_engines/fp16/1-gpu-192-batch/ --max_batch_size 192
trtllm-build --checkpoint_dir gpt2-large/trt_ckpt/fp16/1-gpu/ --gemm_plugin float16 --output_dir gpt2-large/trt_engines/fp16/1-gpu-256-batch/ --max_batch_size 256

# /dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm
sudo docker cp benchmarking/benchmark_trtllm.py b209d39f9c48:/app/tensorrt_llm/examples/benchmark_trtllm.py
sudo docker cp benchmarking/benchmark_utils.py b209d39f9c48:/app/tensorrt_llm/examples/benchmark_utils.py
sudo docker cp ShareGPT_V3_unfiltered_cleaned_split.json b209d39f9c48:/app/tensorrt_llm/examples/ShareGPT_V3_unfiltered_cleaned_split.json
sudo docker cp ShareGPT_V3_unfiltered_cleaned_split_top100.json b209d39f9c48:/app/tensorrt_llm/examples/ShareGPT_V3_unfiltered_cleaned_split_top100.json


# experiments TODO: is this a fair comparison against Llama2? max sequence length for gpt2 is 1024
# 1 gpu 1 batch 500 max
# /dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm
python benchmarking/nvsmi_monitor.py --output_dir outputs/gpt/812M/fp16/1-gpu-1-batch --output_file nvsmi_numreqsample0_iter100_max500_v10032gb.out --container_id b209d39f9c48 --container_output_dir /app/tensorrt_llm/examples/gpt --container_stop_file container_stop.txt --gpu_type v10032gb
# no prompt formatting and extra padding token
# container "/app/tensorrt_llm/examples/gpt"
python ../benchmark_trtllm.py --tokenizer_dir gpt2 --engine_dir gpt2-large/trt_engines/fp16/1-gpu-1-batch --dataset_path ../ShareGPT_V3_unfiltered_cleaned_split.json --num_requests_sample 0 --max_batch_size 1 --max_input_tokens 500 --max_output_tokens 500 --output_dir /app/tensorrt_llm/examples/gpt/outputs/gpt/812M/fp16/1-gpu-1-batch --output_file bmark_numreqsample0_iter100_max500_v10032gb.out --container_output_dir /app/tensorrt_llm/examples/gpt --container_stop_file container_stop.txt --random_seed 42 --num_iterations 100
# /dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm
sudo docker cp b209d39f9c48:/app/tensorrt_llm/examples/gpt/outputs/gpt/812M/fp16/1-gpu-1-batch/bmark_numreqsample0_iter100_max500_v10032gb.out outputs/gpt/812M/fp16/1-gpu-1-batch/bmark_numreqsample0_iter100_max500_v10032gb.out

# 1 gpu 128 batch 500 max
# /dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm
python benchmarking/nvsmi_monitor.py --output_dir outputs/gpt/812M/fp16/1-gpu-128-batch --output_file nvsmi_numreqsample0_iter100_max500_v10032gb.out --container_id b209d39f9c48 --container_output_dir /app/tensorrt_llm/examples/gpt --container_stop_file container_stop.txt --gpu_type v10032gb
# no prompt formatting and extra padding token
# container "/app/tensorrt_llm/examples/gpt"
python ../benchmark_trtllm.py --tokenizer_dir gpt2 --engine_dir gpt2-large/trt_engines/fp16/1-gpu-128-batch --dataset_path ../ShareGPT_V3_unfiltered_cleaned_split.json --num_requests_sample 0 --max_batch_size 128 --max_input_tokens 500 --max_output_tokens 500 --output_dir /app/tensorrt_llm/examples/gpt/outputs/gpt/812M/fp16/1-gpu-128-batch --output_file bmark_numreqsample0_iter100_max500_v10032gb.out --container_output_dir /app/tensorrt_llm/examples/gpt --container_stop_file container_stop.txt --random_seed 42 --num_iterations 100 --no_token_logging
# /dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm
sudo docker cp b209d39f9c48:/app/tensorrt_llm/examples/gpt/outputs/gpt/812M/fp16/1-gpu-128-batch/bmark_numreqsample0_iter100_max500_v10032gb.out outputs/gpt/812M/fp16/1-gpu-128-batch/bmark_numreqsample0_iter100_max500_v10032gb.out


# TODO: OOM
# 1 gpu 136 batch 500 max
# /dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm
python benchmarking/nvsmi_monitor.py --output_dir outputs/gpt/812M/fp16/1-gpu-136-batch --output_file nvsmi_numreqsample0_iter100_max500_v10032gb.out --container_id b209d39f9c48 --container_output_dir /app/tensorrt_llm/examples/gpt --container_stop_file container_stop.txt --gpu_type v10032gb
# no prompt formatting and extra padding token
# container "/app/tensorrt_llm/examples/gpt"
python ../benchmark_trtllm.py --tokenizer_dir gpt2 --engine_dir gpt2-large/trt_engines/fp16/1-gpu-136-batch --dataset_path ../ShareGPT_V3_unfiltered_cleaned_split.json --num_requests_sample 0 --max_batch_size 136 --max_input_tokens 500 --max_output_tokens 500 --output_dir /app/tensorrt_llm/examples/gpt/outputs/gpt/812M/fp16/1-gpu-136-batch --output_file bmark_numreqsample0_iter100_max500_v10032gb.out --container_output_dir /app/tensorrt_llm/examples/gpt --container_stop_file container_stop.txt --random_seed 42 --num_iterations 100 --no_token_logging
# /dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm
sudo docker cp b209d39f9c48:/app/tensorrt_llm/examples/gpt/outputs/gpt/812M/fp16/1-gpu-136-batch/bmark_numreqsample0_iter100_max500_v10032gb.out outputs/gpt/812M/fp16/1-gpu-136-batch/bmark_numreqsample0_iter100_max500_v10032gb.out

# TODO: OOM
# 1 gpu 144 batch 500 max
# /dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm
python benchmarking/nvsmi_monitor.py --output_dir outputs/gpt/812M/fp16/1-gpu-144-batch --output_file nvsmi_numreqsample0_iter100_max500_v10032gb.out --container_id b209d39f9c48 --container_output_dir /app/tensorrt_llm/examples/gpt --container_stop_file container_stop.txt --gpu_type v10032gb
# no prompt formatting and extra padding token
# container "/app/tensorrt_llm/examples/gpt"
python ../benchmark_trtllm.py --tokenizer_dir gpt2 --engine_dir gpt2-large/trt_engines/fp16/1-gpu-144-batch --dataset_path ../ShareGPT_V3_unfiltered_cleaned_split.json --num_requests_sample 0 --max_batch_size 144 --max_input_tokens 500 --max_output_tokens 500 --output_dir /app/tensorrt_llm/examples/gpt/outputs/gpt/812M/fp16/1-gpu-144-batch --output_file bmark_numreqsample0_iter100_max500_v10032gb.out --container_output_dir /app/tensorrt_llm/examples/gpt --container_stop_file container_stop.txt --random_seed 42 --num_iterations 100 --no_token_logging
# /dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm
sudo docker cp b209d39f9c48:/app/tensorrt_llm/examples/gpt/outputs/gpt/812M/fp16/1-gpu-144-batch/bmark_numreqsample0_iter100_max500_v10032gb.out outputs/gpt/812M/fp16/1-gpu-144-batch/bmark_numreqsample0_iter100_max500_v10032gb.out

# TODO: OOM
# 1 gpu 160 batch 500 max
# /dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm
python benchmarking/nvsmi_monitor.py --output_dir outputs/gpt/812M/fp16/1-gpu-160-batch --output_file nvsmi_numreqsample0_iter100_max500_v10032gb.out --container_id b209d39f9c48 --container_output_dir /app/tensorrt_llm/examples/gpt --container_stop_file container_stop.txt --gpu_type v10032gb
# no prompt formatting and extra padding token
# container "/app/tensorrt_llm/examples/gpt"
python ../benchmark_trtllm.py --tokenizer_dir gpt2 --engine_dir gpt2-large/trt_engines/fp16/1-gpu-160-batch --dataset_path ../ShareGPT_V3_unfiltered_cleaned_split.json --num_requests_sample 0 --max_batch_size 160 --max_input_tokens 500 --max_output_tokens 500 --output_dir /app/tensorrt_llm/examples/gpt/outputs/gpt/812M/fp16/1-gpu-160-batch --output_file bmark_numreqsample0_iter100_max500_v10032gb.out --container_output_dir /app/tensorrt_llm/examples/gpt --container_stop_file container_stop.txt --random_seed 42 --num_iterations 100 --no_token_logging
# /dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm
sudo docker cp b209d39f9c48:/app/tensorrt_llm/examples/gpt/outputs/gpt/812M/fp16/1-gpu-160-batch/bmark_numreqsample0_iter100_max500_v10032gb.out outputs/gpt/812M/fp16/1-gpu-160-batch/bmark_numreqsample0_iter100_max500_v10032gb.out

# TODO: OOM
# 1 gpu 192 batch 500 max
# /dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm
python benchmarking/nvsmi_monitor.py --output_dir outputs/gpt/812M/fp16/1-gpu-192-batch --output_file nvsmi_numreqsample0_iter100_max500_v10032gb.out --container_id b209d39f9c48 --container_output_dir /app/tensorrt_llm/examples/gpt --container_stop_file container_stop.txt --gpu_type v10032gb
# no prompt formatting and extra padding token
# container "/app/tensorrt_llm/examples/gpt"
python ../benchmark_trtllm.py --tokenizer_dir gpt2 --engine_dir gpt2-large/trt_engines/fp16/1-gpu-192-batch --dataset_path ../ShareGPT_V3_unfiltered_cleaned_split.json --num_requests_sample 0 --max_batch_size 192 --max_input_tokens 500 --max_output_tokens 500 --output_dir /app/tensorrt_llm/examples/gpt/outputs/gpt/812M/fp16/1-gpu-192-batch --output_file bmark_numreqsample0_iter100_max500_v10032gb.out --container_output_dir /app/tensorrt_llm/examples/gpt --container_stop_file container_stop.txt --random_seed 42 --num_iterations 100 --no_token_logging
# /dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm
sudo docker cp b209d39f9c48:/app/tensorrt_llm/examples/gpt/outputs/gpt/812M/fp16/1-gpu-192-batch/bmark_numreqsample0_iter100_max500_v10032gb.out outputs/gpt/812M/fp16/1-gpu-192-batch/bmark_numreqsample0_iter100_max500_v10032gb.out

# TODO: OOM
# 1 gpu 256 batch 500 max
# /dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm
python benchmarking/nvsmi_monitor.py --output_dir outputs/gpt/812M/fp16/1-gpu-256-batch --output_file nvsmi_numreqsample0_iter100_max500_v10032gb.out --container_id b209d39f9c48 --container_output_dir /app/tensorrt_llm/examples/gpt --container_stop_file container_stop.txt --gpu_type v10032gb
# no prompt formatting and extra padding token
# container "/app/tensorrt_llm/examples/gpt"
python ../benchmark_trtllm.py --tokenizer_dir gpt2 --engine_dir gpt2-large/trt_engines/fp16/1-gpu-256-batch --dataset_path ../ShareGPT_V3_unfiltered_cleaned_split.json --num_requests_sample 0 --max_batch_size 256 --max_input_tokens 500 --max_output_tokens 500 --output_dir /app/tensorrt_llm/examples/gpt/outputs/gpt/812M/fp16/1-gpu-256-batch --output_file bmark_numreqsample0_iter100_max500_v10032gb.out --container_output_dir /app/tensorrt_llm/examples/gpt --container_stop_file container_stop.txt --random_seed 42 --num_iterations 100 --no_token_logging
# /dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm
sudo docker cp b209d39f9c48:/app/tensorrt_llm/examples/gpt/outputs/gpt/812M/fp16/1-gpu-256-batch/bmark_numreqsample0_iter100_max500_v10032gb.out outputs/gpt/812M/fp16/1-gpu-256-batch/bmark_numreqsample0_iter100_max500_v10032gb.out
