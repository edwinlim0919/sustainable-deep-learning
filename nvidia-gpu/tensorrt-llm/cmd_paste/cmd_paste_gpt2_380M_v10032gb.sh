# TensorRT-LLM standalone


### MISCELLANEOUS HELPER COMMANDS
# replace docker container id in this cmd_paste with the current one
sed -i "s/75005fc8bfdc/75005fc8bfdc/g" cmd_paste_gpt2_380M_v10032gb.sh

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
rm -rf gpt2-medium && git clone https://huggingface.co/gpt2-medium gpt2-medium
pushd gpt2-medium && rm pytorch_model.bin model.safetensors && wget -q https://huggingface.co/gpt2-medium/resolve/main/pytorch_model.bin && popd

# 1-gpu fp16 nowq
python3 convert_checkpoint.py --model_dir gpt2-medium --dtype float16 --output_dir gpt2-medium/trt_ckpt/fp16/1-gpu/
trtllm-build --checkpoint_dir gpt2-medium/trt_ckpt/fp16/1-gpu/ --gemm_plugin float16 --output_dir gpt2-medium/trt_engines/fp16/1-gpu-1-batch/ --max_batch_size 1
trtllm-build --checkpoint_dir gpt2-medium/trt_ckpt/fp16/1-gpu/ --gemm_plugin float16 --output_dir gpt2-medium/trt_engines/fp16/1-gpu-128-batch/ --max_batch_size 128
trtllm-build --checkpoint_dir gpt2-medium/trt_ckpt/fp16/1-gpu/ --gemm_plugin float16 --output_dir gpt2-medium/trt_engines/fp16/1-gpu-256-batch/ --max_batch_size 256
trtllm-build --checkpoint_dir gpt2-medium/trt_ckpt/fp16/1-gpu/ --gemm_plugin float16 --output_dir gpt2-medium/trt_engines/fp16/1-gpu-384-batch/ --max_batch_size 384
trtllm-build --checkpoint_dir gpt2-medium/trt_ckpt/fp16/1-gpu/ --gemm_plugin float16 --output_dir gpt2-medium/trt_engines/fp16/1-gpu-512-batch/ --max_batch_size 512
trtllm-build --checkpoint_dir gpt2-medium/trt_ckpt/fp16/1-gpu/ --gemm_plugin float16 --output_dir gpt2-medium/trt_engines/fp16/1-gpu-640-batch/ --max_batch_size 640
trtllm-build --checkpoint_dir gpt2-medium/trt_ckpt/fp16/1-gpu/ --gemm_plugin float16 --output_dir gpt2-medium/trt_engines/fp16/1-gpu-768-batch/ --max_batch_size 768
trtllm-build --checkpoint_dir gpt2-medium/trt_ckpt/fp16/1-gpu/ --gemm_plugin float16 --output_dir gpt2-medium/trt_engines/fp16/1-gpu-896-batch/ --max_batch_size 896
trtllm-build --checkpoint_dir gpt2-medium/trt_ckpt/fp16/1-gpu/ --gemm_plugin float16 --output_dir gpt2-medium/trt_engines/fp16/1-gpu-928-batch/ --max_batch_size 928

# OOM
trtllm-build --checkpoint_dir gpt2-medium/trt_ckpt/fp16/1-gpu/ --gemm_plugin float16 --output_dir gpt2-medium/trt_engines/fp16/1-gpu-960-batch/ --max_batch_size 960
trtllm-build --checkpoint_dir gpt2-medium/trt_ckpt/fp16/1-gpu/ --gemm_plugin float16 --output_dir gpt2-medium/trt_engines/fp16/1-gpu-1024-batch/ --max_batch_size 1024

# /dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm
sudo docker cp ../benchmarking/benchmark_trtllm.py 75005fc8bfdc:/TensorRT-LLM/examples/benchmark_trtllm.py
sudo docker cp ../benchmarking/benchmark_utils.py 75005fc8bfdc:/TensorRT-LLM/examples/benchmark_utils.py
sudo docker cp ../ShareGPT_V3_unfiltered_cleaned_split.json 75005fc8bfdc:/TensorRT-LLM/examples/ShareGPT_V3_unfiltered_cleaned_split.json
sudo docker cp ../ShareGPT_V3_unfiltered_cleaned_split_top100.json 75005fc8bfdc:/TensorRT-LLM/examples/ShareGPT_V3_unfiltered_cleaned_split_top100.json


### EXPERIMENT COMMANDS
# fp16 nowq batch 1
python3 benchmarking/nvsmi_monitor.py --output_dir outputs/gpt/380M/fp16/1-gpu-1-batch --output_file nvsmi_numreqsample0_iter100_max500_v10032gb.out --container_id 75005fc8bfdc --container_output_dir /TensorRT-LLM/examples/gpt --container_stop_file container_stop.txt --gpu_type v10032gb
python3 ../benchmark_trtllm.py --tokenizer_dir gpt2-medium --engine_dir gpt2-medium/trt_engines/fp16/1-gpu-1-batch --dataset_path ../ShareGPT_V3_unfiltered_cleaned_split.json --num_requests_sample 0 --max_batch_size 1 --max_input_tokens 500 --max_output_tokens 500 --output_dir /TensorRT-LLM/examples/gpt/outputs/gpt/380M/fp16/1-gpu-1-batch --output_file bmark_numreqsample0_iter100_max500_v10032gb.out --container_output_dir /TensorRT-LLM/examples/gpt --container_stop_file container_stop.txt --random_seed 42 --num_iterations 100 --no_token_logging
sudo docker cp 75005fc8bfdc:/TensorRT-LLM/examples/gpt/outputs/gpt/380M/fp16/1-gpu-1-batch/bmark_numreqsample0_iter100_max500_v10032gb.out ../outputs/gpt/380M/fp16/1-gpu-1-batch/bmark_numreqsample0_iter100_max500_v10032gb.out

# fp16 nowq batch 128
python3 benchmarking/nvsmi_monitor.py --output_dir outputs/gpt/380M/fp16/1-gpu-128-batch --output_file nvsmi_numreqsample0_iter100_max500_v10032gb.out --container_id 75005fc8bfdc --container_output_dir /TensorRT-LLM/examples/gpt --container_stop_file container_stop.txt --gpu_type v10032gb
python3 ../benchmark_trtllm.py --tokenizer_dir gpt2-medium --engine_dir gpt2-medium/trt_engines/fp16/1-gpu-128-batch --dataset_path ../ShareGPT_V3_unfiltered_cleaned_split.json --num_requests_sample 0 --max_batch_size 128 --max_input_tokens 500 --max_output_tokens 500 --output_dir /TensorRT-LLM/examples/gpt/outputs/gpt/380M/fp16/1-gpu-128-batch --output_file bmark_numreqsample0_iter100_max500_v10032gb.out --container_output_dir /TensorRT-LLM/examples/gpt --container_stop_file container_stop.txt --random_seed 42 --num_iterations 100 --no_token_logging
sudo docker cp 75005fc8bfdc:/TensorRT-LLM/examples/gpt/outputs/gpt/380M/fp16/1-gpu-128-batch/bmark_numreqsample0_iter100_max500_v10032gb.out ../outputs/gpt/380M/fp16/1-gpu-128-batch/bmark_numreqsample0_iter100_max500_v10032gb.out

# fp16 nowq batch 256
python3 benchmarking/nvsmi_monitor.py --output_dir outputs/gpt/380M/fp16/1-gpu-256-batch --output_file nvsmi_numreqsample0_iter100_max500_v10032gb.out --container_id 75005fc8bfdc --container_output_dir /TensorRT-LLM/examples/gpt --container_stop_file container_stop.txt --gpu_type v10032gb
python3 ../benchmark_trtllm.py --tokenizer_dir gpt2-medium --engine_dir gpt2-medium/trt_engines/fp16/1-gpu-256-batch --dataset_path ../ShareGPT_V3_unfiltered_cleaned_split.json --num_requests_sample 0 --max_batch_size 256 --max_input_tokens 500 --max_output_tokens 500 --output_dir /TensorRT-LLM/examples/gpt/outputs/gpt/380M/fp16/1-gpu-256-batch --output_file bmark_numreqsample0_iter100_max500_v10032gb.out --container_output_dir /TensorRT-LLM/examples/gpt --container_stop_file container_stop.txt --random_seed 42 --num_iterations 100 --no_token_logging
sudo docker cp 75005fc8bfdc:/TensorRT-LLM/examples/gpt/outputs/gpt/380M/fp16/1-gpu-256-batch/bmark_numreqsample0_iter100_max500_v10032gb.out ../outputs/gpt/380M/fp16/1-gpu-256-batch/bmark_numreqsample0_iter100_max500_v10032gb.out

# fp16 nowq batch 384
python3 benchmarking/nvsmi_monitor.py --output_dir outputs/gpt/380M/fp16/1-gpu-384-batch --output_file nvsmi_numreqsample0_iter100_max500_v10032gb.out --container_id 75005fc8bfdc --container_output_dir /TensorRT-LLM/examples/gpt --container_stop_file container_stop.txt --gpu_type v10032gb
python3 ../benchmark_trtllm.py --tokenizer_dir gpt2-medium --engine_dir gpt2-medium/trt_engines/fp16/1-gpu-384-batch --dataset_path ../ShareGPT_V3_unfiltered_cleaned_split.json --num_requests_sample 0 --max_batch_size 384 --max_input_tokens 500 --max_output_tokens 500 --output_dir /TensorRT-LLM/examples/gpt/outputs/gpt/380M/fp16/1-gpu-384-batch --output_file bmark_numreqsample0_iter100_max500_v10032gb.out --container_output_dir /TensorRT-LLM/examples/gpt --container_stop_file container_stop.txt --random_seed 42 --num_iterations 100 --no_token_logging
sudo docker cp 75005fc8bfdc:/TensorRT-LLM/examples/gpt/outputs/gpt/380M/fp16/1-gpu-384-batch/bmark_numreqsample0_iter100_max500_v10032gb.out ../outputs/gpt/380M/fp16/1-gpu-384-batch/bmark_numreqsample0_iter100_max500_v10032gb.out

# fp16 nowq batch 512
python3 benchmarking/nvsmi_monitor.py --output_dir outputs/gpt/380M/fp16/1-gpu-512-batch --output_file nvsmi_numreqsample0_iter100_max500_v10032gb.out --container_id 75005fc8bfdc --container_output_dir /TensorRT-LLM/examples/gpt --container_stop_file container_stop.txt --gpu_type v10032gb
python3 ../benchmark_trtllm.py --tokenizer_dir gpt2-medium --engine_dir gpt2-medium/trt_engines/fp16/1-gpu-512-batch --dataset_path ../ShareGPT_V3_unfiltered_cleaned_split.json --num_requests_sample 0 --max_batch_size 512 --max_input_tokens 500 --max_output_tokens 500 --output_dir /TensorRT-LLM/examples/gpt/outputs/gpt/380M/fp16/1-gpu-512-batch --output_file bmark_numreqsample0_iter100_max500_v10032gb.out --container_output_dir /TensorRT-LLM/examples/gpt --container_stop_file container_stop.txt --random_seed 42 --num_iterations 100 --no_token_logging
sudo docker cp 75005fc8bfdc:/TensorRT-LLM/examples/gpt/outputs/gpt/380M/fp16/1-gpu-512-batch/bmark_numreqsample0_iter100_max500_v10032gb.out ../outputs/gpt/380M/fp16/1-gpu-512-batch/bmark_numreqsample0_iter100_max500_v10032gb.out

# fp16 nowq batch 640
python3 benchmarking/nvsmi_monitor.py --output_dir outputs/gpt/380M/fp16/1-gpu-640-batch --output_file nvsmi_numreqsample0_iter100_max500_v10032gb.out --container_id 75005fc8bfdc --container_output_dir /TensorRT-LLM/examples/gpt --container_stop_file container_stop.txt --gpu_type v10032gb
python3 ../benchmark_trtllm.py --tokenizer_dir gpt2-medium --engine_dir gpt2-medium/trt_engines/fp16/1-gpu-640-batch --dataset_path ../ShareGPT_V3_unfiltered_cleaned_split.json --num_requests_sample 0 --max_batch_size 640 --max_input_tokens 500 --max_output_tokens 500 --output_dir /TensorRT-LLM/examples/gpt/outputs/gpt/380M/fp16/1-gpu-640-batch --output_file bmark_numreqsample0_iter100_max500_v10032gb.out --container_output_dir /TensorRT-LLM/examples/gpt --container_stop_file container_stop.txt --random_seed 42 --num_iterations 100 --no_token_logging
sudo docker cp 75005fc8bfdc:/TensorRT-LLM/examples/gpt/outputs/gpt/380M/fp16/1-gpu-640-batch/bmark_numreqsample0_iter100_max500_v10032gb.out ../outputs/gpt/380M/fp16/1-gpu-640-batch/bmark_numreqsample0_iter100_max500_v10032gb.out

# fp16 nowq batch 768
python3 benchmarking/nvsmi_monitor.py --output_dir outputs/gpt/380M/fp16/1-gpu-768-batch --output_file nvsmi_numreqsample0_iter100_max500_v10032gb.out --container_id 75005fc8bfdc --container_output_dir /TensorRT-LLM/examples/gpt --container_stop_file container_stop.txt --gpu_type v10032gb
python3 ../benchmark_trtllm.py --tokenizer_dir gpt2-medium --engine_dir gpt2-medium/trt_engines/fp16/1-gpu-768-batch --dataset_path ../ShareGPT_V3_unfiltered_cleaned_split.json --num_requests_sample 0 --max_batch_size 768 --max_input_tokens 500 --max_output_tokens 500 --output_dir /TensorRT-LLM/examples/gpt/outputs/gpt/380M/fp16/1-gpu-768-batch --output_file bmark_numreqsample0_iter100_max500_v10032gb.out --container_output_dir /TensorRT-LLM/examples/gpt --container_stop_file container_stop.txt --random_seed 42 --num_iterations 100 --no_token_logging
sudo docker cp 75005fc8bfdc:/TensorRT-LLM/examples/gpt/outputs/gpt/380M/fp16/1-gpu-768-batch/bmark_numreqsample0_iter100_max500_v10032gb.out ../outputs/gpt/380M/fp16/1-gpu-768-batch/bmark_numreqsample0_iter100_max500_v10032gb.out

# fp16 nowq batch 896
python3 benchmarking/nvsmi_monitor.py --output_dir outputs/gpt/380M/fp16/1-gpu-896-batch --output_file nvsmi_numreqsample0_iter100_max500_v10032gb.out --container_id 75005fc8bfdc --container_output_dir /TensorRT-LLM/examples/gpt --container_stop_file container_stop.txt --gpu_type v10032gb
python3 ../benchmark_trtllm.py --tokenizer_dir gpt2-medium --engine_dir gpt2-medium/trt_engines/fp16/1-gpu-896-batch --dataset_path ../ShareGPT_V3_unfiltered_cleaned_split.json --num_requests_sample 0 --max_batch_size 896 --max_input_tokens 500 --max_output_tokens 500 --output_dir /TensorRT-LLM/examples/gpt/outputs/gpt/380M/fp16/1-gpu-896-batch --output_file bmark_numreqsample0_iter100_max500_v10032gb.out --container_output_dir /TensorRT-LLM/examples/gpt --container_stop_file container_stop.txt --random_seed 42 --num_iterations 100 --no_token_logging
sudo docker cp 75005fc8bfdc:/TensorRT-LLM/examples/gpt/outputs/gpt/380M/fp16/1-gpu-896-batch/bmark_numreqsample0_iter100_max500_v10032gb.out ../outputs/gpt/380M/fp16/1-gpu-896-batch/bmark_numreqsample0_iter100_max500_v10032gb.out

# fp16 nowq batch 928
python3 benchmarking/nvsmi_monitor.py --output_dir outputs/gpt/380M/fp16/1-gpu-928-batch --output_file nvsmi_numreqsample0_iter100_max500_v10032gb.out --container_id 75005fc8bfdc --container_output_dir /TensorRT-LLM/examples/gpt --container_stop_file container_stop.txt --gpu_type v10032gb
python3 ../benchmark_trtllm.py --tokenizer_dir gpt2-medium --engine_dir gpt2-medium/trt_engines/fp16/1-gpu-928-batch --dataset_path ../ShareGPT_V3_unfiltered_cleaned_split.json --num_requests_sample 0 --max_batch_size 928 --max_input_tokens 500 --max_output_tokens 500 --output_dir /TensorRT-LLM/examples/gpt/outputs/gpt/380M/fp16/1-gpu-928-batch --output_file bmark_numreqsample0_iter100_max500_v10032gb.out --container_output_dir /TensorRT-LLM/examples/gpt --container_stop_file container_stop.txt --random_seed 42 --num_iterations 100 --no_token_logging
sudo docker cp 75005fc8bfdc:/TensorRT-LLM/examples/gpt/outputs/gpt/380M/fp16/1-gpu-928-batch/bmark_numreqsample0_iter100_max500_v10032gb.out ../outputs/gpt/380M/fp16/1-gpu-928-batch/bmark_numreqsample0_iter100_max500_v10032gb.out

# OOM
# fp16 nowq batch 960
python3 benchmarking/nvsmi_monitor.py --output_dir outputs/gpt/380M/fp16/1-gpu-960-batch --output_file nvsmi_numreqsample0_iter100_max500_v10032gb.out --container_id 75005fc8bfdc --container_output_dir /TensorRT-LLM/examples/gpt --container_stop_file container_stop.txt --gpu_type v10032gb
python3 ../benchmark_trtllm.py --tokenizer_dir gpt2-medium --engine_dir gpt2-medium/trt_engines/fp16/1-gpu-960-batch --dataset_path ../ShareGPT_V3_unfiltered_cleaned_split.json --num_requests_sample 0 --max_batch_size 960 --max_input_tokens 500 --max_output_tokens 500 --output_dir /TensorRT-LLM/examples/gpt/outputs/gpt/380M/fp16/1-gpu-960-batch --output_file bmark_numreqsample0_iter100_max500_v10032gb.out --container_output_dir /TensorRT-LLM/examples/gpt --container_stop_file container_stop.txt --random_seed 42 --num_iterations 100 --no_token_logging
sudo docker cp 75005fc8bfdc:/TensorRT-LLM/examples/gpt/outputs/gpt/380M/fp16/1-gpu-960-batch/bmark_numreqsample0_iter100_max500_v10032gb.out ../outputs/gpt/380M/fp16/1-gpu-960-batch/bmark_numreqsample0_iter100_max500_v10032gb.out

# OOM
# fp16 nowq batch 1024
python3 benchmarking/nvsmi_monitor.py --output_dir outputs/gpt/380M/fp16/1-gpu-1024-batch --output_file nvsmi_numreqsample0_iter100_max500_v10032gb.out --container_id 75005fc8bfdc --container_output_dir /TensorRT-LLM/examples/gpt --container_stop_file container_stop.txt --gpu_type v10032gb
python3 ../benchmark_trtllm.py --tokenizer_dir gpt2-medium --engine_dir gpt2-medium/trt_engines/fp16/1-gpu-1024-batch --dataset_path ../ShareGPT_V3_unfiltered_cleaned_split.json --num_requests_sample 0 --max_batch_size 1024 --max_input_tokens 500 --max_output_tokens 500 --output_dir /TensorRT-LLM/examples/gpt/outputs/gpt/380M/fp16/1-gpu-1024-batch --output_file bmark_numreqsample0_iter100_max500_v10032gb.out --container_output_dir /TensorRT-LLM/examples/gpt --container_stop_file container_stop.txt --random_seed 42 --num_iterations 100 --no_token_logging
sudo docker cp 75005fc8bfdc:/TensorRT-LLM/examples/gpt/outputs/gpt/380M/fp16/1-gpu-1024-batch/bmark_numreqsample0_iter100_max500_v10032gb.out ../outputs/gpt/380M/fp16/1-gpu-1024-batch/bmark_numreqsample0_iter100_max500_v10032gb.out
