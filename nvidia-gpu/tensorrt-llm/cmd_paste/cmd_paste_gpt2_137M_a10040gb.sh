# TensorRT-LLM standalone


### MISCELLANEOUS HELPER COMMANDS
# replace docker container id in this cmd_paste with the current one
sed -i "s/dec7beecb269/dec7beecb269/g" cmd_paste_gpt2_137M_a10040gb.sh

# flushing GPU memory
sudo fuser -v /dev/nvidia*
sudo kill -9 <PID>

# disable MIG if Pytorch cannot find CUDA
nvidia-smi
sudo nvidia-smi -mig 0
nvidia-smi


### GPT2 SETUP COMMANDS
# container "/TensorRT-LLM/examples/gpt"
pip3 install -r requirements.txt
rm -rf gpt2 && git clone https://huggingface.co/gpt2 gpt2
pushd gpt2 && rm pytorch_model.bin model.safetensors && wget -q https://huggingface.co/gpt2/resolve/main/pytorch_model.bin && popd

# 1-gpu fp16 nowq
python3 convert_checkpoint.py --model_dir gpt2 --dtype float16 --output_dir gpt2/trt_ckpt/fp16/1-gpu/
trtllm-build --checkpoint_dir gpt2/trt_ckpt/fp16/1-gpu/ --gemm_plugin float16 --output_dir gpt2/trt_engines/fp16/1-gpu-1-batch/ --max_batch_size 1
trtllm-build --checkpoint_dir gpt2/trt_ckpt/fp16/1-gpu/ --gemm_plugin float16 --output_dir gpt2/trt_engines/fp16/1-gpu-128-batch/ --max_batch_size 128
trtllm-build --checkpoint_dir gpt2/trt_ckpt/fp16/1-gpu/ --gemm_plugin float16 --output_dir gpt2/trt_engines/fp16/1-gpu-256-batch/ --max_batch_size 256
trtllm-build --checkpoint_dir gpt2/trt_ckpt/fp16/1-gpu/ --gemm_plugin float16 --output_dir gpt2/trt_engines/fp16/1-gpu-384-batch/ --max_batch_size 384
trtllm-build --checkpoint_dir gpt2/trt_ckpt/fp16/1-gpu/ --gemm_plugin float16 --output_dir gpt2/trt_engines/fp16/1-gpu-512-batch/ --max_batch_size 512
trtllm-build --checkpoint_dir gpt2/trt_ckpt/fp16/1-gpu/ --gemm_plugin float16 --output_dir gpt2/trt_engines/fp16/1-gpu-640-batch/ --max_batch_size 640
trtllm-build --checkpoint_dir gpt2/trt_ckpt/fp16/1-gpu/ --gemm_plugin float16 --output_dir gpt2/trt_engines/fp16/1-gpu-768-batch/ --max_batch_size 768
trtllm-build --checkpoint_dir gpt2/trt_ckpt/fp16/1-gpu/ --gemm_plugin float16 --output_dir gpt2/trt_engines/fp16/1-gpu-896-batch/ --max_batch_size 896
trtllm-build --checkpoint_dir gpt2/trt_ckpt/fp16/1-gpu/ --gemm_plugin float16 --output_dir gpt2/trt_engines/fp16/1-gpu-1024-batch/ --max_batch_size 1024
trtllm-build --checkpoint_dir gpt2/trt_ckpt/fp16/1-gpu/ --gemm_plugin float16 --output_dir gpt2/trt_engines/fp16/1-gpu-1152-batch/ --max_batch_size 1152
trtllm-build --checkpoint_dir gpt2/trt_ckpt/fp16/1-gpu/ --gemm_plugin float16 --output_dir gpt2/trt_engines/fp16/1-gpu-1280-batch/ --max_batch_size 1280
trtllm-build --checkpoint_dir gpt2/trt_ckpt/fp16/1-gpu/ --gemm_plugin float16 --output_dir gpt2/trt_engines/fp16/1-gpu-1344-batch/ --max_batch_size 1344

# OOM
trtllm-build --checkpoint_dir gpt2/trt_ckpt/fp16/1-gpu/ --gemm_plugin float16 --output_dir gpt2/trt_engines/fp16/1-gpu-1408-batch/ --max_batch_size 1408

# /dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm
sudo docker cp ../benchmarking/benchmark_trtllm.py dec7beecb269:/TensorRT-LLM/examples/benchmark_trtllm.py
sudo docker cp ../benchmarking/benchmark_utils.py dec7beecb269:/TensorRT-LLM/examples/benchmark_utils.py
sudo docker cp ../ShareGPT_V3_unfiltered_cleaned_split.json dec7beecb269:/TensorRT-LLM/examples/ShareGPT_V3_unfiltered_cleaned_split.json
sudo docker cp ../ShareGPT_V3_unfiltered_cleaned_split_top100.json dec7beecb269:/TensorRT-LLM/examples/ShareGPT_V3_unfiltered_cleaned_split_top100.json


### DEV TESTING
python3 benchmarking/nvsmi_monitor.py --output_dir outputs/gpt/137M/fp16/1-gpu-1-batch --output_file nvsmi_dev_testing.out --container_id dec7beecb269 --container_output_dir /TensorRT-LLM/examples/gpt --container_stop_file container_stop.txt --gpu_type a10040gb
python3 ../benchmark_trtllm.py --tokenizer_dir gpt2 --engine_dir gpt2/trt_engines/fp16/1-gpu-1-batch --dataset_path ../ShareGPT_V3_unfiltered_cleaned_split.json --num_requests_sample 0 --max_batch_size 1 --max_input_tokens 500 --max_output_tokens 500 --output_dir /TensorRT-LLM/examples/gpt/outputs/gpt/137M/fp16/1-gpu-1-batch --output_file bmark_dev_testing.out --container_output_dir /TensorRT-LLM/examples/gpt --container_stop_file container_stop.txt --random_seed 42 --num_iterations 10 --no_token_logging
sudo docker cp dec7beecb269:/TensorRT-LLM/examples/gpt/outputs/gpt/137M/fp16/1-gpu-1-batch/bmark_dev_testing.out ../outputs/gpt/137M/fp16/1-gpu-1-batch/bmark_dev_testing.out


### EXPERIMENT COMMANDS
# fp16 nowq batch 1
python3 benchmarking/nvsmi_monitor.py --output_dir outputs/gpt/137M/fp16/1-gpu-1-batch --output_file nvsmi_numreqsample0_iter100_max500_a10040gb.out --container_id dec7beecb269 --container_output_dir /TensorRT-LLM/examples/gpt --container_stop_file container_stop.txt --gpu_type a10040gb
python3 ../benchmark_trtllm.py --tokenizer_dir gpt2 --engine_dir gpt2/trt_engines/fp16/1-gpu-1-batch --dataset_path ../ShareGPT_V3_unfiltered_cleaned_split.json --num_requests_sample 0 --max_batch_size 1 --max_input_tokens 500 --max_output_tokens 500 --output_dir /TensorRT-LLM/examples/gpt/outputs/gpt/137M/fp16/1-gpu-1-batch --output_file bmark_numreqsample0_iter100_max500_a10040gb.out --container_output_dir /TensorRT-LLM/examples/gpt --container_stop_file container_stop.txt --random_seed 42 --num_iterations 100 --no_token_logging
sudo docker cp dec7beecb269:/TensorRT-LLM/examples/gpt/outputs/gpt/137M/fp16/1-gpu-1-batch/bmark_numreqsample0_iter100_max500_a10040gb.out ../outputs/gpt/137M/fp16/1-gpu-1-batch/bmark_numreqsample0_iter100_max500_a10040gb.out

# fp16 nowq batch 128
python3 benchmarking/nvsmi_monitor.py --output_dir outputs/gpt/137M/fp16/1-gpu-128-batch --output_file nvsmi_numreqsample0_iter100_max500_a10040gb.out --container_id dec7beecb269 --container_output_dir /TensorRT-LLM/examples/gpt --container_stop_file container_stop.txt --gpu_type a10040gb
python3 ../benchmark_trtllm.py --tokenizer_dir gpt2 --engine_dir gpt2/trt_engines/fp16/1-gpu-128-batch --dataset_path ../ShareGPT_V3_unfiltered_cleaned_split.json --num_requests_sample 0 --max_batch_size 128 --max_input_tokens 500 --max_output_tokens 500 --output_dir /TensorRT-LLM/examples/gpt/outputs/gpt/137M/fp16/1-gpu-128-batch --output_file bmark_numreqsample0_iter100_max500_a10040gb.out --container_output_dir /TensorRT-LLM/examples/gpt --container_stop_file container_stop.txt --random_seed 42 --num_iterations 100 --no_token_logging
sudo docker cp dec7beecb269:/TensorRT-LLM/examples/gpt/outputs/gpt/137M/fp16/1-gpu-128-batch/bmark_numreqsample0_iter100_max500_a10040gb.out ../outputs/gpt/137M/fp16/1-gpu-128-batch/bmark_numreqsample0_iter100_max500_a10040gb.out

# fp16 nowq batch 256
python3 benchmarking/nvsmi_monitor.py --output_dir outputs/gpt/137M/fp16/1-gpu-256-batch --output_file nvsmi_numreqsample0_iter100_max500_a10040gb.out --container_id dec7beecb269 --container_output_dir /TensorRT-LLM/examples/gpt --container_stop_file container_stop.txt --gpu_type a10040gb
python3 ../benchmark_trtllm.py --tokenizer_dir gpt2 --engine_dir gpt2/trt_engines/fp16/1-gpu-256-batch --dataset_path ../ShareGPT_V3_unfiltered_cleaned_split.json --num_requests_sample 0 --max_batch_size 256 --max_input_tokens 500 --max_output_tokens 500 --output_dir /TensorRT-LLM/examples/gpt/outputs/gpt/137M/fp16/1-gpu-256-batch --output_file bmark_numreqsample0_iter100_max500_a10040gb.out --container_output_dir /TensorRT-LLM/examples/gpt --container_stop_file container_stop.txt --random_seed 42 --num_iterations 100 --no_token_logging
sudo docker cp dec7beecb269:/TensorRT-LLM/examples/gpt/outputs/gpt/137M/fp16/1-gpu-256-batch/bmark_numreqsample0_iter100_max500_a10040gb.out ../outputs/gpt/137M/fp16/1-gpu-256-batch/bmark_numreqsample0_iter100_max500_a10040gb.out

# fp16 nowq batch 384
python3 benchmarking/nvsmi_monitor.py --output_dir outputs/gpt/137M/fp16/1-gpu-384-batch --output_file nvsmi_numreqsample0_iter100_max500_a10040gb.out --container_id dec7beecb269 --container_output_dir /TensorRT-LLM/examples/gpt --container_stop_file container_stop.txt --gpu_type a10040gb
python3 ../benchmark_trtllm.py --tokenizer_dir gpt2 --engine_dir gpt2/trt_engines/fp16/1-gpu-384-batch --dataset_path ../ShareGPT_V3_unfiltered_cleaned_split.json --num_requests_sample 0 --max_batch_size 384 --max_input_tokens 500 --max_output_tokens 500 --output_dir /TensorRT-LLM/examples/gpt/outputs/gpt/137M/fp16/1-gpu-384-batch --output_file bmark_numreqsample0_iter100_max500_a10040gb.out --container_output_dir /TensorRT-LLM/examples/gpt --container_stop_file container_stop.txt --random_seed 42 --num_iterations 100 --no_token_logging
sudo docker cp dec7beecb269:/TensorRT-LLM/examples/gpt/outputs/gpt/137M/fp16/1-gpu-384-batch/bmark_numreqsample0_iter100_max500_a10040gb.out ../outputs/gpt/137M/fp16/1-gpu-384-batch/bmark_numreqsample0_iter100_max500_a10040gb.out

# fp16 nowq batch 512
python3 benchmarking/nvsmi_monitor.py --output_dir outputs/gpt/137M/fp16/1-gpu-512-batch --output_file nvsmi_numreqsample0_iter100_max500_a10040gb.out --container_id dec7beecb269 --container_output_dir /TensorRT-LLM/examples/gpt --container_stop_file container_stop.txt --gpu_type a10040gb
python3 ../benchmark_trtllm.py --tokenizer_dir gpt2 --engine_dir gpt2/trt_engines/fp16/1-gpu-512-batch --dataset_path ../ShareGPT_V3_unfiltered_cleaned_split.json --num_requests_sample 0 --max_batch_size 512 --max_input_tokens 500 --max_output_tokens 500 --output_dir /TensorRT-LLM/examples/gpt/outputs/gpt/137M/fp16/1-gpu-512-batch --output_file bmark_numreqsample0_iter100_max500_a10040gb.out --container_output_dir /TensorRT-LLM/examples/gpt --container_stop_file container_stop.txt --random_seed 42 --num_iterations 100 --no_token_logging
sudo docker cp dec7beecb269:/TensorRT-LLM/examples/gpt/outputs/gpt/137M/fp16/1-gpu-512-batch/bmark_numreqsample0_iter100_max500_a10040gb.out ../outputs/gpt/137M/fp16/1-gpu-512-batch/bmark_numreqsample0_iter100_max500_a10040gb.out

# fp16 nowq batch 640
python3 benchmarking/nvsmi_monitor.py --output_dir outputs/gpt/137M/fp16/1-gpu-640-batch --output_file nvsmi_numreqsample0_iter100_max500_a10040gb.out --container_id dec7beecb269 --container_output_dir /TensorRT-LLM/examples/gpt --container_stop_file container_stop.txt --gpu_type a10040gb
python3 ../benchmark_trtllm.py --tokenizer_dir gpt2 --engine_dir gpt2/trt_engines/fp16/1-gpu-640-batch --dataset_path ../ShareGPT_V3_unfiltered_cleaned_split.json --num_requests_sample 0 --max_batch_size 640 --max_input_tokens 500 --max_output_tokens 500 --output_dir /TensorRT-LLM/examples/gpt/outputs/gpt/137M/fp16/1-gpu-640-batch --output_file bmark_numreqsample0_iter100_max500_a10040gb.out --container_output_dir /TensorRT-LLM/examples/gpt --container_stop_file container_stop.txt --random_seed 42 --num_iterations 100 --no_token_logging
sudo docker cp dec7beecb269:/TensorRT-LLM/examples/gpt/outputs/gpt/137M/fp16/1-gpu-640-batch/bmark_numreqsample0_iter100_max500_a10040gb.out ../outputs/gpt/137M/fp16/1-gpu-640-batch/bmark_numreqsample0_iter100_max500_a10040gb.out

# fp16 nowq batch 768
python3 benchmarking/nvsmi_monitor.py --output_dir outputs/gpt/137M/fp16/1-gpu-768-batch --output_file nvsmi_numreqsample0_iter100_max500_a10040gb.out --container_id dec7beecb269 --container_output_dir /TensorRT-LLM/examples/gpt --container_stop_file container_stop.txt --gpu_type a10040gb
python3 ../benchmark_trtllm.py --tokenizer_dir gpt2 --engine_dir gpt2/trt_engines/fp16/1-gpu-768-batch --dataset_path ../ShareGPT_V3_unfiltered_cleaned_split.json --num_requests_sample 0 --max_batch_size 768 --max_input_tokens 500 --max_output_tokens 500 --output_dir /TensorRT-LLM/examples/gpt/outputs/gpt/137M/fp16/1-gpu-768-batch --output_file bmark_numreqsample0_iter100_max500_a10040gb.out --container_output_dir /TensorRT-LLM/examples/gpt --container_stop_file container_stop.txt --random_seed 42 --num_iterations 100 --no_token_logging
sudo docker cp dec7beecb269:/TensorRT-LLM/examples/gpt/outputs/gpt/137M/fp16/1-gpu-768-batch/bmark_numreqsample0_iter100_max500_a10040gb.out ../outputs/gpt/137M/fp16/1-gpu-768-batch/bmark_numreqsample0_iter100_max500_a10040gb.out

# fp16 nowq batch 896
python3 benchmarking/nvsmi_monitor.py --output_dir outputs/gpt/137M/fp16/1-gpu-896-batch --output_file nvsmi_numreqsample0_iter100_max500_a10040gb.out --container_id dec7beecb269 --container_output_dir /TensorRT-LLM/examples/gpt --container_stop_file container_stop.txt --gpu_type a10040gb
python3 ../benchmark_trtllm.py --tokenizer_dir gpt2 --engine_dir gpt2/trt_engines/fp16/1-gpu-896-batch --dataset_path ../ShareGPT_V3_unfiltered_cleaned_split.json --num_requests_sample 0 --max_batch_size 896 --max_input_tokens 500 --max_output_tokens 500 --output_dir /TensorRT-LLM/examples/gpt/outputs/gpt/137M/fp16/1-gpu-896-batch --output_file bmark_numreqsample0_iter100_max500_a10040gb.out --container_output_dir /TensorRT-LLM/examples/gpt --container_stop_file container_stop.txt --random_seed 42 --num_iterations 100 --no_token_logging
sudo docker cp dec7beecb269:/TensorRT-LLM/examples/gpt/outputs/gpt/137M/fp16/1-gpu-896-batch/bmark_numreqsample0_iter100_max500_a10040gb.out ../outputs/gpt/137M/fp16/1-gpu-896-batch/bmark_numreqsample0_iter100_max500_a10040gb.out

# fp16 nowq batch 1024
python3 benchmarking/nvsmi_monitor.py --output_dir outputs/gpt/137M/fp16/1-gpu-1024-batch --output_file nvsmi_numreqsample0_iter100_max500_a10040gb.out --container_id dec7beecb269 --container_output_dir /TensorRT-LLM/examples/gpt --container_stop_file container_stop.txt --gpu_type a10040gb
python3 ../benchmark_trtllm.py --tokenizer_dir gpt2 --engine_dir gpt2/trt_engines/fp16/1-gpu-1024-batch --dataset_path ../ShareGPT_V3_unfiltered_cleaned_split.json --num_requests_sample 0 --max_batch_size 1024 --max_input_tokens 500 --max_output_tokens 500 --output_dir /TensorRT-LLM/examples/gpt/outputs/gpt/137M/fp16/1-gpu-1024-batch --output_file bmark_numreqsample0_iter100_max500_a10040gb.out --container_output_dir /TensorRT-LLM/examples/gpt --container_stop_file container_stop.txt --random_seed 42 --num_iterations 100 --no_token_logging
sudo docker cp dec7beecb269:/TensorRT-LLM/examples/gpt/outputs/gpt/137M/fp16/1-gpu-1024-batch/bmark_numreqsample0_iter100_max500_a10040gb.out ../outputs/gpt/137M/fp16/1-gpu-1024-batch/bmark_numreqsample0_iter100_max500_a10040gb.out

# fp16 nowq batch 1152
python3 benchmarking/nvsmi_monitor.py --output_dir outputs/gpt/137M/fp16/1-gpu-1152-batch --output_file nvsmi_numreqsample0_iter100_max500_a10040gb.out --container_id dec7beecb269 --container_output_dir /TensorRT-LLM/examples/gpt --container_stop_file container_stop.txt --gpu_type a10040gb
python3 ../benchmark_trtllm.py --tokenizer_dir gpt2 --engine_dir gpt2/trt_engines/fp16/1-gpu-1152-batch --dataset_path ../ShareGPT_V3_unfiltered_cleaned_split.json --num_requests_sample 0 --max_batch_size 1152 --max_input_tokens 500 --max_output_tokens 500 --output_dir /TensorRT-LLM/examples/gpt/outputs/gpt/137M/fp16/1-gpu-1152-batch --output_file bmark_numreqsample0_iter100_max500_a10040gb.out --container_output_dir /TensorRT-LLM/examples/gpt --container_stop_file container_stop.txt --random_seed 42 --num_iterations 100 --no_token_logging
sudo docker cp dec7beecb269:/TensorRT-LLM/examples/gpt/outputs/gpt/137M/fp16/1-gpu-1152-batch/bmark_numreqsample0_iter100_max500_a10040gb.out ../outputs/gpt/137M/fp16/1-gpu-1152-batch/bmark_numreqsample0_iter100_max500_a10040gb.out

# fp16 nowq batch 1280
python3 benchmarking/nvsmi_monitor.py --output_dir outputs/gpt/137M/fp16/1-gpu-1280-batch --output_file nvsmi_numreqsample0_iter100_max500_a10040gb.out --container_id dec7beecb269 --container_output_dir /TensorRT-LLM/examples/gpt --container_stop_file container_stop.txt --gpu_type a10040gb
python3 ../benchmark_trtllm.py --tokenizer_dir gpt2 --engine_dir gpt2/trt_engines/fp16/1-gpu-1280-batch --dataset_path ../ShareGPT_V3_unfiltered_cleaned_split.json --num_requests_sample 0 --max_batch_size 1280 --max_input_tokens 500 --max_output_tokens 500 --output_dir /TensorRT-LLM/examples/gpt/outputs/gpt/137M/fp16/1-gpu-1280-batch --output_file bmark_numreqsample0_iter100_max500_a10040gb.out --container_output_dir /TensorRT-LLM/examples/gpt --container_stop_file container_stop.txt --random_seed 42 --num_iterations 100 --no_token_logging
sudo docker cp dec7beecb269:/TensorRT-LLM/examples/gpt/outputs/gpt/137M/fp16/1-gpu-1280-batch/bmark_numreqsample0_iter100_max500_a10040gb.out ../outputs/gpt/137M/fp16/1-gpu-1280-batch/bmark_numreqsample0_iter100_max500_a10040gb.out

# fp16 nowq batch 1344
python3 benchmarking/nvsmi_monitor.py --output_dir outputs/gpt/137M/fp16/1-gpu-1344-batch --output_file nvsmi_numreqsample0_iter100_max500_a10040gb.out --container_id dec7beecb269 --container_output_dir /TensorRT-LLM/examples/gpt --container_stop_file container_stop.txt --gpu_type a10040gb
python3 ../benchmark_trtllm.py --tokenizer_dir gpt2 --engine_dir gpt2/trt_engines/fp16/1-gpu-1344-batch --dataset_path ../ShareGPT_V3_unfiltered_cleaned_split.json --num_requests_sample 0 --max_batch_size 1344 --max_input_tokens 500 --max_output_tokens 500 --output_dir /TensorRT-LLM/examples/gpt/outputs/gpt/137M/fp16/1-gpu-1344-batch --output_file bmark_numreqsample0_iter100_max500_a10040gb.out --container_output_dir /TensorRT-LLM/examples/gpt --container_stop_file container_stop.txt --random_seed 42 --num_iterations 100 --no_token_logging
sudo docker cp dec7beecb269:/TensorRT-LLM/examples/gpt/outputs/gpt/137M/fp16/1-gpu-1344-batch/bmark_numreqsample0_iter100_max500_a10040gb.out ../outputs/gpt/137M/fp16/1-gpu-1344-batch/bmark_numreqsample0_iter100_max500_a10040gb.out



# OOM
# fp16 nowq batch 1408
python3 benchmarking/nvsmi_monitor.py --output_dir outputs/gpt/137M/fp16/1-gpu-1408-batch --output_file nvsmi_numreqsample0_iter100_max500_a10040gb.out --container_id dec7beecb269 --container_output_dir /TensorRT-LLM/examples/gpt --container_stop_file container_stop.txt --gpu_type a10040gb
python3 ../benchmark_trtllm.py --tokenizer_dir gpt2 --engine_dir gpt2/trt_engines/fp16/1-gpu-1408-batch --dataset_path ../ShareGPT_V3_unfiltered_cleaned_split.json --num_requests_sample 0 --max_batch_size 1408 --max_input_tokens 500 --max_output_tokens 500 --output_dir /TensorRT-LLM/examples/gpt/outputs/gpt/137M/fp16/1-gpu-1408-batch --output_file bmark_numreqsample0_iter100_max500_a10040gb.out --container_output_dir /TensorRT-LLM/examples/gpt --container_stop_file container_stop.txt --random_seed 42 --num_iterations 100 --no_token_logging
sudo docker cp dec7beecb269:/TensorRT-LLM/examples/gpt/outputs/gpt/137M/fp16/1-gpu-1408-batch/bmark_numreqsample0_iter100_max500_a10040gb.out ../outputs/gpt/137M/fp16/1-gpu-1407-batch/bmark_numreqsample0_iter100_max500_a10040gb.out
