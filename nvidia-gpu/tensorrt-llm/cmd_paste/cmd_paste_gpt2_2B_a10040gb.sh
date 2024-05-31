# TensorRT-LLM standalone


# HELPER COMMANDS
sed -i "s/e20fe0c7f878/e20fe0c7f878/g" cmd_paste_gpt2_2B_a10040gb.sh # replace docker container id in cmd_paste.sh with the current one

# Flushing GPU memory
sudo fuser -v /dev/nvidia*
sudo kill -9 <PID>


# GPT2 SETUP COMMANDS
# container "/TensorRT-LLM/examples/gpt"
pip install -r requirements.txt
rm -rf gpt2-xl && git clone https://huggingface.co/gpt2-xl gpt2-xl
pushd gpt2-xl && rm pytorch_model.bin model.safetensors && wget -q https://huggingface.co/gpt2-xl/resolve/main/pytorch_model.bin && popd

# 1-gpu fp16 nowq
python3 convert_checkpoint.py --model_dir gpt2-xl --dtype float16 --output_dir gpt2-xl/trt_ckpt/fp16/1-gpu/
trtllm-build --checkpoint_dir gpt2-xl/trt_ckpt/fp16/1-gpu/ --gemm_plugin float16 --output_dir gpt2-xl/trt_engines/fp16/1-gpu-1-batch/ --max_batch_size 1
trtllm-build --checkpoint_dir gpt2-xl/trt_ckpt/fp16/1-gpu/ --gemm_plugin float16 --output_dir gpt2-xl/trt_engines/fp16/1-gpu-256-batch/ --max_batch_size 256

# /dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm
sudo docker cp benchmarking/benchmark_trtllm.py e20fe0c7f878:/TensorRT-LLM/examples/benchmark_trtllm.py
sudo docker cp benchmarking/benchmark_utils.py e20fe0c7f878:/TensorRT-LLM/examples/benchmark_utils.py
sudo docker cp ShareGPT_V3_unfiltered_cleaned_split.json e20fe0c7f878:/TensorRT-LLM/examples/ShareGPT_V3_unfiltered_cleaned_split.json
sudo docker cp ShareGPT_V3_unfiltered_cleaned_split_top100.json e20fe0c7f878:/TensorRT-LLM/examples/ShareGPT_V3_unfiltered_cleaned_split_top100.json


# EXPERIMENT COMMANDS
# 1 gpu 1 batch 500 max
python3 benchmarking/nvsmi_monitor.py --output_dir outputs/gpt/2B/fp16/1-gpu-1-batch --output_file nvsmi_numreqsample0_iter100_max500_a10040gb.out --container_id e20fe0c7f878 --container_output_dir /TensorRT-LLM/examples/gpt --container_stop_file container_stop.txt --gpu_type a10040gb
python3 ../benchmark_trtllm.py --tokenizer_dir gpt2-xl --engine_dir gpt2-xl/trt_engines/fp16/1-gpu-1-batch --dataset_path ../ShareGPT_V3_unfiltered_cleaned_split.json --num_requests_sample 0 --max_batch_size 1 --max_input_tokens 500 --max_output_tokens 500 --output_dir /TensorRT-LLM/examples/gpt/outputs/gpt/2B/fp16/1-gpu-1-batch --output_file bmark_numreqsample0_iter100_max500_a10040gb.out --container_output_dir /TensorRT-LLM/examples/gpt --container_stop_file container_stop.txt --random_seed 42 --num_iterations 100 --no_token_logging
sudo docker cp e20fe0c7f878:/TensorRT-LLM/examples/gpt/outputs/gpt/2B/fp16/1-gpu-1-batch/bmark_numreqsample0_iter100_max500_a10040gb.out ../outputs/gpt/2B/fp16/1-gpu-1-batch/bmark_numreqsample0_iter100_max500_a10040gb.out

# 1 gpu 256 batch 500 max
python3 benchmarking/nvsmi_monitor.py --output_dir outputs/gpt/2B/fp16/1-gpu-256-batch --output_file nvsmi_numreqsample0_iter100_max500_a10040gb.out --container_id e20fe0c7f878 --container_output_dir /TensorRT-LLM/examples/gpt --container_stop_file container_stop.txt --gpu_type a10040gb
python3 ../benchmark_trtllm.py --tokenizer_dir gpt2-xl --engine_dir gpt2-xl/trt_engines/fp16/1-gpu-256-batch --dataset_path ../ShareGPT_V3_unfiltered_cleaned_split.json --num_requests_sample 0 --max_batch_size 256 --max_input_tokens 500 --max_output_tokens 500 --output_dir /TensorRT-LLM/examples/gpt/outputs/gpt/2B/fp16/1-gpu-256-batch --output_file bmark_numreqsample0_iter100_max500_a10040gb.out --container_output_dir /TensorRT-LLM/examples/gpt --container_stop_file container_stop.txt --random_seed 42 --num_iterations 100 --no_token_logging
sudo docker cp e20fe0c7f878:/TensorRT-LLM/examples/gpt/outputs/gpt/2B/fp16/1-gpu-256-batch/bmark_numreqsample0_iter100_max500_a10040gb.out ../outputs/gpt/2B/fp16/1-gpu-256-batch/bmark_numreqsample0_iter100_max500_a10040gb.out
