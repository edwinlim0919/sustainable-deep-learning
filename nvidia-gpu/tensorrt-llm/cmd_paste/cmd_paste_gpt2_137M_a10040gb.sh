# TensorRT-LLM standalone


### HELPER COMMANDS
# replace docker container id in this cmd_paste with the current one
sed -i "s/668b65a92f55/668b65a92f55/g" cmd_paste_gpt2_137M_a10040gb.sh

# flushing GPU memory
sudo fuser -v /dev/nvidia*
sudo kill -9 <PID>

# disable MIG if Pytorch cannot find CUDA
nvidia-smi
sudo nvidia-smi -mig 0
nvidia-smi


### GPT2 SETUP COMMANDS
# container "/TensorRT-LLM/examples/gpt"
pip install -r requirements.txt
rm -rf gpt2 && git clone https://huggingface.co/gpt2 gpt2
pushd gpt2 && rm pytorch_model.bin model.safetensors && wget -q https://huggingface.co/gpt2/resolve/main/pytorch_model.bin && popd

# 1-gpu fp16 nowq
python3 convert_checkpoint.py --model_dir gpt2 --dtype float16 --output_dir gpt2/trt_ckpt/fp16/1-gpu/
trtllm-build --checkpoint_dir gpt2/trt_ckpt/fp16/1-gpu/ --gemm_plugin float16 --output_dir gpt2/trt_engines/fp16/1-gpu-1-batch/ --max_batch_size 1

# /dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm
sudo docker cp benchmarking/benchmark_trtllm.py 668b65a92f55:/app/tensorrt_llm/examples/benchmark_trtllm.py
sudo docker cp benchmarking/benchmark_utils.py 668b65a92f55:/app/tensorrt_llm/examples/benchmark_utils.py
sudo docker cp ShareGPT_V3_unfiltered_cleaned_split.json 668b65a92f55:/app/tensorrt_llm/examples/ShareGPT_V3_unfiltered_cleaned_split.json
sudo docker cp ShareGPT_V3_unfiltered_cleaned_split_top100.json 668b65a92f55:/app/tensorrt_llm/examples/ShareGPT_V3_unfiltered_cleaned_split_top100.json


### EXPERIMENT COMMANDS
# TODO: is this a fair comparison against Llama2? max sequence length for gpt2 is 1024
# 1 gpu 1 batch 500 max
# /dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm
python benchmarking/nvsmi_monitor.py --output_dir outputs/gpt/137M/fp16/1-gpu-1-batch --output_file nvsmi_numreqsample0_iter100_max500_v10032gb.out --container_id 668b65a92f55 --container_output_dir /app/tensorrt_llm/examples/gpt --container_stop_file container_stop.txt --gpu_type v10032gb
# no prompt formatting and extra padding token
python ../benchmark_trtllm.py --tokenizer_dir gpt2 --engine_dir gpt2/trt_engines/fp16/1-gpu-1-batch --dataset_path ../ShareGPT_V3_unfiltered_cleaned_split.json --num_requests_sample 0 --max_batch_size 1 --max_input_tokens 500 --max_output_tokens 500 --output_dir /app/tensorrt_llm/examples/gpt/outputs/gpt/137M/fp16/1-gpu-1-batch --output_file bmark_numreqsample0_iter100_max500_v10032gb.out --container_output_dir /app/tensorrt_llm/examples/gpt --container_stop_file container_stop.txt --random_seed 42 --num_iterations 100
# /dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm
sudo docker cp 668b65a92f55:/app/tensorrt_llm/examples/gpt/outputs/gpt/137M/fp16/1-gpu-1-batch/bmark_numreqsample0_iter100_max500_v10032gb.out outputs/gpt/137M/fp16/1-gpu-1-batch/bmark_numreqsample0_iter100_max500_v10032gb.out
