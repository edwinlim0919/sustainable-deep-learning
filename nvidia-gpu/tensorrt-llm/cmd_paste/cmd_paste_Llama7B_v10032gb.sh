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
sed -i "s/5742c720375f/c71f5b2d5461/g" cmd_paste_Llama7B.sh # replace docker container id in cmd_paste.sh with the current one

conda activate tensorrt-llm
huggingface-cli login
python3 download_hf_weights.py --model-name "meta-llama/Llama-2-7b-chat-hf"
sudo docker cp meta-llama/ c71f5b2d5461:/app/tensorrt_llm/examples/llama

# /app/tensorrt_llm/examples/llama
pip install nltk
pip install rouge_score
pip install aiofiles

# Non-quantized
python convert_checkpoint.py --model_dir meta-llama/Llama-2-7b-chat-hf_model --dtype float16 --output_dir ./llama/7B/trt_ckpt/fp16/1-gpu/
trtllm-build --checkpoint_dir ./llama/7B/trt_ckpt/fp16/1-gpu/ --gemm_plugin float16 --output_dir ./llama/7B/trt_engines/fp16/1-gpu-1-batch/ --max_batch_size 1
trtllm-build --checkpoint_dir ./llama/7B/trt_ckpt/fp16/1-gpu/ --gemm_plugin float16 --output_dir ./llama/7B/trt_engines/fp16/1-gpu-2-batch/ --max_batch_size 2
trtllm-build --checkpoint_dir ./llama/7B/trt_ckpt/fp16/1-gpu/ --gemm_plugin float16 --output_dir ./llama/7B/trt_engines/fp16/1-gpu-4-batch/ --max_batch_size 4
trtllm-build --checkpoint_dir ./llama/7B/trt_ckpt/fp16/1-gpu/ --gemm_plugin float16 --output_dir ./llama/7B/trt_engines/fp16/1-gpu-6-batch/ --max_batch_size 6
trtllm-build --checkpoint_dir ./llama/7B/trt_ckpt/fp16/1-gpu/ --gemm_plugin float16 --output_dir ./llama/7B/trt_engines/fp16/1-gpu-8-batch/ --max_batch_size 8
trtllm-build --checkpoint_dir ./llama/7B/trt_ckpt/fp16/1-gpu/ --gemm_plugin float16 --output_dir ./llama/7B/trt_engines/fp16/1-gpu-10-batch/ --max_batch_size 10
trtllm-build --checkpoint_dir ./llama/7B/trt_ckpt/fp16/1-gpu/ --gemm_plugin float16 --output_dir ./llama/7B/trt_engines/fp16/1-gpu-12-batch/ --max_batch_size 12
trtllm-build --checkpoint_dir ./llama/7B/trt_ckpt/fp16/1-gpu/ --gemm_plugin float16 --output_dir ./llama/7B/trt_engines/fp16/1-gpu-14-batch/ --max_batch_size 14
trtllm-build --checkpoint_dir ./llama/7B/trt_ckpt/fp16/1-gpu/ --gemm_plugin float16 --output_dir ./llama/7B/trt_engines/fp16/1-gpu-16-batch/ --max_batch_size 16
trtllm-build --checkpoint_dir ./llama/7B/trt_ckpt/fp16/1-gpu/ --gemm_plugin float16 --output_dir ./llama/7B/trt_engines/fp16/1-gpu-18-batch/ --max_batch_size 18
trtllm-build --checkpoint_dir ./llama/7B/trt_ckpt/fp16/1-gpu/ --gemm_plugin float16 --output_dir ./llama/7B/trt_engines/fp16/1-gpu-20-batch/ --max_batch_size 20
trtllm-build --checkpoint_dir ./llama/7B/trt_ckpt/fp16/1-gpu/ --gemm_plugin float16 --output_dir ./llama/7B/trt_engines/fp16/1-gpu-22-batch/ --max_batch_size 22
trtllm-build --checkpoint_dir ./llama/7B/trt_ckpt/fp16/1-gpu/ --gemm_plugin float16 --output_dir ./llama/7B/trt_engines/fp16/1-gpu-24-batch/ --max_batch_size 24
trtllm-build --checkpoint_dir ./llama/7B/trt_ckpt/fp16/1-gpu/ --gemm_plugin float16 --output_dir ./llama/7B/trt_engines/fp16/1-gpu-26-batch/ --max_batch_size 26
trtllm-build --checkpoint_dir ./llama/7B/trt_ckpt/fp16/1-gpu/ --gemm_plugin float16 --output_dir ./llama/7B/trt_engines/fp16/1-gpu-28-batch/ --max_batch_size 28
trtllm-build --checkpoint_dir ./llama/7B/trt_ckpt/fp16/1-gpu/ --gemm_plugin float16 --output_dir ./llama/7B/trt_engines/fp16/1-gpu-30-batch/ --max_batch_size 30
trtllm-build --checkpoint_dir ./llama/7B/trt_ckpt/fp16/1-gpu/ --gemm_plugin float16 --output_dir ./llama/7B/trt_engines/fp16/1-gpu-32-batch/ --max_batch_size 32
trtllm-build --checkpoint_dir ./llama/7B/trt_ckpt/fp16/1-gpu/ --gemm_plugin float16 --output_dir ./llama/7B/trt_engines/fp16/1-gpu-64-batch/ --max_batch_size 64

# TODO: 2 changes to plotting/parsing scripts and reformatting existing results
#       - multiple dictionaries in nvsmi results for each GPU index
#       - output token lengths no longer in length-1 list
#       - adding name change for v100 results
# 4-bit weight-quantized
python convert_checkpoint.py --model_dir meta-llama/Llama-2-7b-chat-hf_model --output_dir ./llama/7B/trt_ckpt/fp16_wq4/1-gpu/ --dtype float16 --use_weight_only --weight_only_precision int4
trtllm-build --checkpoint_dir ./llama/7B/trt_ckpt/fp16_wq4/1-gpu/ --gemm_plugin float16 --output_dir ./llama/7B/trt_engines/fp16_wq4/1-gpu-1-batch/ --max_batch_size 1
trtllm-build --checkpoint_dir ./llama/7B/trt_ckpt/fp16_wq4/1-gpu/ --gemm_plugin float16 --output_dir ./llama/7B/trt_engines/fp16_wq4/1-gpu-4-batch/ --max_batch_size 4
trtllm-build --checkpoint_dir ./llama/7B/trt_ckpt/fp16_wq4/1-gpu/ --gemm_plugin float16 --output_dir ./llama/7B/trt_engines/fp16_wq4/1-gpu-8-batch/ --max_batch_size 8
trtllm-build --checkpoint_dir ./llama/7B/trt_ckpt/fp16_wq4/1-gpu/ --gemm_plugin float16 --output_dir ./llama/7B/trt_engines/fp16_wq4/1-gpu-12-batch/ --max_batch_size 12
trtllm-build --checkpoint_dir ./llama/7B/trt_ckpt/fp16_wq4/1-gpu/ --gemm_plugin float16 --output_dir ./llama/7B/trt_engines/fp16_wq4/1-gpu-16-batch/ --max_batch_size 16
trtllm-build --checkpoint_dir ./llama/7B/trt_ckpt/fp16_wq4/1-gpu/ --gemm_plugin float16 --output_dir ./llama/7B/trt_engines/fp16_wq4/1-gpu-20-batch/ --max_batch_size 20
trtllm-build --checkpoint_dir ./llama/7B/trt_ckpt/fp16_wq4/1-gpu/ --gemm_plugin float16 --output_dir ./llama/7B/trt_engines/fp16_wq4/1-gpu-22-batch/ --max_batch_size 22
trtllm-build --checkpoint_dir ./llama/7B/trt_ckpt/fp16_wq4/1-gpu/ --gemm_plugin float16 --output_dir ./llama/7B/trt_engines/fp16_wq4/1-gpu-24-batch/ --max_batch_size 24


# /dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm
sudo docker cp benchmarking/benchmark_trtllm.py c71f5b2d5461:/app/tensorrt_llm/examples/benchmark_trtllm.py
sudo docker cp benchmarking/benchmark_utils.py c71f5b2d5461:/app/tensorrt_llm/examples/benchmark_utils.py
sudo docker cp ShareGPT_V3_unfiltered_cleaned_split.json c71f5b2d5461:/app/tensorrt_llm/examples/ShareGPT_V3_unfiltered_cleaned_split.json
sudo docker cp ShareGPT_V3_unfiltered_cleaned_split_top100.json c71f5b2d5461:/app/tensorrt_llm/examples/ShareGPT_V3_unfiltered_cleaned_split_top100.json

# /app/tensorrt_llm/examples/llama
# dev testing
python ../benchmark_trtllm.py --tokenizer_dir ./meta-llama/Llama-2-7b-chat-hf_tokenizer/ --engine_dir ./llama/7B/trt_engines/fp16/1-gpu-1-batch/ --dataset_path ../ShareGPT_V3_unfiltered_cleaned_split_top100.json --num_requests_sample 4 --max_batch_size 1 --max_input_tokens 1000 --max_output_tokens 1000 --output_dir ./outputs/llama/7B/fp16/1-gpu-1-batch --output_file dev_testing.out --random_seed 42 --num_iterations 10
python ../benchmark_trtllm.py --tokenizer_dir ./meta-llama/Llama-2-7b-chat-hf_tokenizer/ --engine_dir ./llama/7B/trt_engines/fp16/1-gpu-2-batch/ --dataset_path ../ShareGPT_V3_unfiltered_cleaned_split_top100.json --num_requests_sample 4 --max_batch_size 2 --max_input_tokens 1000 --max_output_tokens 1000 --output_dir ./outputs/llama/7B/fp16/1-gpu-2-batch --output_file dev_testing.out --random_seed 42 --num_iterations 1

# experiments

# 1 gpu 1 batch
# /dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm
python benchmarking/nvsmi_monitor.py --output_dir outputs/llama/7B/fp16/1-gpu-1-batch --output_file nvsmi_numreqsample10000_iter100.out
# /app/tensorrt_llm/examples/llama
python ../benchmark_trtllm.py --tokenizer_dir ./meta-llama/Llama-2-7b-chat-hf_tokenizer/ --engine_dir ./llama/7B/trt_engines/fp16/1-gpu-1-batch/ --dataset_path ../ShareGPT_V3_unfiltered_cleaned_split.json --num_requests_sample 10000 --max_batch_size 1 --max_input_tokens 1000 --max_output_tokens 1000 --output_dir ./outputs/llama/7B/fp16/1-gpu-1-batch --output_file bmark_numreqsample10000_iter100.out --random_seed 42 --num_iterations 100
# /dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm
sudo docker cp c71f5b2d5461:/app/tensorrt_llm/examples/llama/outputs/llama/7B/fp16/1-gpu-1-batch/bmark_numreqsample10000_iter100.out outputs/llama/7B/fp16/1-gpu-1-batch/bmark_numreqsample10000_iter100.out

# 1 gpu 2 batch
# /dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm
python benchmarking/nvsmi_monitor.py --output_dir outputs/llama/7B/fp16/1-gpu-2-batch --output_file nvsmi_numreqsample10000_iter100.out
# /app/tensorrt_llm/examples/llama
python ../benchmark_trtllm.py --tokenizer_dir ./meta-llama/Llama-2-7b-chat-hf_tokenizer/ --engine_dir ./llama/7B/trt_engines/fp16/1-gpu-2-batch/ --dataset_path ../ShareGPT_V3_unfiltered_cleaned_split.json --num_requests_sample 10000 --max_batch_size 2 --max_input_tokens 1000 --max_output_tokens 1000 --output_dir ./outputs/llama/7B/fp16/1-gpu-2-batch --output_file bmark_numreqsample10000_iter100.out --random_seed 42 --num_iterations 100
# /dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm
sudo docker cp c71f5b2d5461:/app/tensorrt_llm/examples/llama/outputs/llama/7B/fp16/1-gpu-2-batch/bmark_numreqsample10000_iter100.out outputs/llama/7B/fp16/1-gpu-2-batch/bmark_numreqsample10000_iter100.out

# 1 gpu 4 batch
# /dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm
python benchmarking/nvsmi_monitor.py --output_dir outputs/llama/7B/fp16/1-gpu-4-batch --output_file nvsmi_numreqsample10000_iter100.out
# /app/tensorrt_llm/examples/llama
python ../benchmark_trtllm.py --tokenizer_dir ./meta-llama/Llama-2-7b-chat-hf_tokenizer/ --engine_dir ./llama/7B/trt_engines/fp16/1-gpu-4-batch/ --dataset_path ../ShareGPT_V3_unfiltered_cleaned_split.json --num_requests_sample 10000 --max_batch_size 4 --max_input_tokens 1000 --max_output_tokens 1000 --output_dir ./outputs/llama/7B/fp16/1-gpu-4-batch --output_file bmark_numreqsample10000_iter100.out --random_seed 42 --num_iterations 100
# /dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm
sudo docker cp c71f5b2d5461:/app/tensorrt_llm/examples/llama/outputs/llama/7B/fp16/1-gpu-4-batch/bmark_numreqsample10000_iter100.out outputs/llama/7B/fp16/1-gpu-4-batch/bmark_numreqsample10000_iter100.out

# 1 gpu 6 batch
# /dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm
python benchmarking/nvsmi_monitor.py --output_dir outputs/llama/7B/fp16/1-gpu-6-batch --output_file nvsmi_numreqsample10000_iter100.out
# /app/tensorrt_llm/examples/llama
python ../benchmark_trtllm.py --tokenizer_dir ./meta-llama/Llama-2-7b-chat-hf_tokenizer/ --engine_dir ./llama/7B/trt_engines/fp16/1-gpu-6-batch/ --dataset_path ../ShareGPT_V3_unfiltered_cleaned_split.json --num_requests_sample 10000 --max_batch_size 6 --max_input_tokens 1000 --max_output_tokens 1000 --output_dir ./outputs/llama/7B/fp16/1-gpu-6-batch --output_file bmark_numreqsample10000_iter100.out --random_seed 42 --num_iterations 100
# /dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm
sudo docker cp c71f5b2d5461:/app/tensorrt_llm/examples/llama/outputs/llama/7B/fp16/1-gpu-6-batch/bmark_numreqsample10000_iter100.out outputs/llama/7B/fp16/1-gpu-6-batch/bmark_numreqsample10000_iter100.out

# 1 gpu 8 batch
# /dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm
python benchmarking/nvsmi_monitor.py --output_dir outputs/llama/7B/fp16/1-gpu-8-batch --output_file nvsmi_numreqsample10000_iter100.out
# /app/tensorrt_llm/examples/llama
python ../benchmark_trtllm.py --tokenizer_dir ./meta-llama/Llama-2-7b-chat-hf_tokenizer/ --engine_dir ./llama/7B/trt_engines/fp16/1-gpu-8-batch/ --dataset_path ../ShareGPT_V3_unfiltered_cleaned_split.json --num_requests_sample 10000 --max_batch_size 8 --max_input_tokens 1000 --max_output_tokens 1000 --output_dir ./outputs/llama/7B/fp16/1-gpu-8-batch --output_file bmark_numreqsample10000_iter100.out --random_seed 42 --num_iterations 100
# /dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm
sudo docker cp c71f5b2d5461:/app/tensorrt_llm/examples/llama/outputs/llama/7B/fp16/1-gpu-8-batch/bmark_numreqsample10000_iter100.out outputs/llama/7B/fp16/1-gpu-8-batch/bmark_numreqsample10000_iter100.out

# 1 gpu 10 batch
# /dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm
python benchmarking/nvsmi_monitor.py --output_dir outputs/llama/7B/fp16/1-gpu-10-batch --output_file nvsmi_numreqsample10000_iter100.out
# /app/tensorrt_llm/examples/llama
python ../benchmark_trtllm.py --tokenizer_dir ./meta-llama/Llama-2-7b-chat-hf_tokenizer/ --engine_dir ./llama/7B/trt_engines/fp16/1-gpu-10-batch/ --dataset_path ../ShareGPT_V3_unfiltered_cleaned_split.json --num_requests_sample 10000 --max_batch_size 10 --max_input_tokens 1000 --max_output_tokens 1000 --output_dir ./outputs/llama/7B/fp16/1-gpu-10-batch --output_file bmark_numreqsample10000_iter100.out --random_seed 42 --num_iterations 100
# /dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm
sudo docker cp c71f5b2d5461:/app/tensorrt_llm/examples/llama/outputs/llama/7B/fp16/1-gpu-10-batch/bmark_numreqsample10000_iter100.out outputs/llama/7B/fp16/1-gpu-10-batch/bmark_numreqsample10000_iter100.out

# 1 gpu 12 batch
# /dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm
python benchmarking/nvsmi_monitor.py --output_dir outputs/llama/7B/fp16/1-gpu-12-batch --output_file nvsmi_numreqsample10000_iter100.out
# /app/tensorrt_llm/examples/llama
python ../benchmark_trtllm.py --tokenizer_dir ./meta-llama/Llama-2-7b-chat-hf_tokenizer/ --engine_dir ./llama/7B/trt_engines/fp16/1-gpu-12-batch/ --dataset_path ../ShareGPT_V3_unfiltered_cleaned_split.json --num_requests_sample 10000 --max_batch_size 12 --max_input_tokens 1000 --max_output_tokens 1000 --output_dir ./outputs/llama/7B/fp16/1-gpu-12-batch --output_file bmark_numreqsample10000_iter100.out --random_seed 42 --num_iterations 100
# /dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm
sudo docker cp c71f5b2d5461:/app/tensorrt_llm/examples/llama/outputs/llama/7B/fp16/1-gpu-12-batch/bmark_numreqsample10000_iter100.out outputs/llama/7B/fp16/1-gpu-12-batch/bmark_numreqsample10000_iter100.out

# 1 gpu 14 batch
# /dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm
python benchmarking/nvsmi_monitor.py --output_dir outputs/llama/7B/fp16/1-gpu-14-batch --output_file nvsmi_numreqsample10000_iter100.out
# /app/tensorrt_llm/examples/llama
python ../benchmark_trtllm.py --tokenizer_dir ./meta-llama/Llama-2-7b-chat-hf_tokenizer/ --engine_dir ./llama/7B/trt_engines/fp16/1-gpu-14-batch/ --dataset_path ../ShareGPT_V3_unfiltered_cleaned_split.json --num_requests_sample 10000 --max_batch_size 14 --max_input_tokens 1000 --max_output_tokens 1000 --output_dir ./outputs/llama/7B/fp16/1-gpu-14-batch --output_file bmark_numreqsample10000_iter100.out --random_seed 42 --num_iterations 100
# /dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm
sudo docker cp c71f5b2d5461:/app/tensorrt_llm/examples/llama/outputs/llama/7B/fp16/1-gpu-14-batch/bmark_numreqsample10000_iter100.out outputs/llama/7B/fp16/1-gpu-14-batch/bmark_numreqsample10000_iter100.out

# 1 gpu 16 batch TODO OOM errors w/ batch size 16
# /dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm
python benchmarking/nvsmi_monitor.py --output_dir outputs/llama/7B/fp16/1-gpu-16-batch --output_file nvsmi_numreqsample10000_iter100.out
# /app/tensorrt_llm/examples/llama
python ../benchmark_trtllm.py --tokenizer_dir ./meta-llama/Llama-2-7b-chat-hf_tokenizer/ --engine_dir ./llama/7B/trt_engines/fp16/1-gpu-16-batch/ --dataset_path ../ShareGPT_V3_unfiltered_cleaned_split.json --num_requests_sample 10000 --max_batch_size 16 --max_input_tokens 1000 --max_output_tokens 1000 --output_dir ./outputs/llama/7B/fp16/1-gpu-16-batch --output_file bmark_numreqsample10000_iter100.out --random_seed 42 --num_iterations 100
# /dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm
sudo docker cp c71f5b2d5461:/app/tensorrt_llm/examples/llama/outputs/llama/7B/fp16/1-gpu-16-batch/bmark_numreqsample10000_iter100.out outputs/llama/7B/fp16/1-gpu-16-batch/bmark_numreqsample10000_iter100.out







# UPDATED EXPERIMENTS
# 1 gpu 1 batch 1000 max
# /dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm
python benchmarking/nvsmi_monitor.py --output_dir outputs/llama/7B/fp16/1-gpu-1-batch --output_file nvsmi_numreqsample0_iter100_max1000.out
# /app/tensorrt_llm/examples/llama
python ../benchmark_trtllm.py --tokenizer_dir ./meta-llama/Llama-2-7b-chat-hf_tokenizer/ --engine_dir ./llama/7B/trt_engines/fp16/1-gpu-1-batch/ --dataset_path ../ShareGPT_V3_unfiltered_cleaned_split.json --num_requests_sample 0 --max_batch_size 1 --max_input_tokens 1000 --max_output_tokens 1000 --output_dir ./outputs/llama/7B/fp16/1-gpu-1-batch --output_file bmark_numreqsample0_iter100_max1000.out --random_seed 42 --num_iterations 100 --use_prompt_formatting --add_special_tokens
# /dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm
sudo docker cp c71f5b2d5461:/app/tensorrt_llm/examples/llama/outputs/llama/7B/fp16/1-gpu-1-batch/bmark_numreqsample0_iter100_max1000.out outputs/llama/7B/fp16/1-gpu-1-batch/bmark_numreqsample0_iter100_max1000.out

# 1 gpu 1 batch 500 max
# /dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm
python benchmarking/nvsmi_monitor.py --output_dir outputs/llama/7B/fp16/1-gpu-1-batch --output_file nvsmi_numreqsample0_iter100_max500.out
# /app/tensorrt_llm/examples/llama
python ../benchmark_trtllm.py --tokenizer_dir ./meta-llama/Llama-2-7b-chat-hf_tokenizer/ --engine_dir ./llama/7B/trt_engines/fp16/1-gpu-1-batch/ --dataset_path ../ShareGPT_V3_unfiltered_cleaned_split.json --num_requests_sample 0 --max_batch_size 1 --max_input_tokens 500 --max_output_tokens 500 --output_dir ./outputs/llama/7B/fp16/1-gpu-1-batch --output_file bmark_numreqsample0_iter100_max500.out --random_seed 42 --num_iterations 100 --use_prompt_formatting --add_special_tokens
# /dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm
sudo docker cp c71f5b2d5461:/app/tensorrt_llm/examples/llama/outputs/llama/7B/fp16/1-gpu-1-batch/bmark_numreqsample0_iter100_max500.out outputs/llama/7B/fp16/1-gpu-1-batch/bmark_numreqsample0_iter100_max500.out

# 1 gpu 2 batch 1000 max
# /dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm
python benchmarking/nvsmi_monitor.py --output_dir outputs/llama/7B/fp16/1-gpu-2-batch --output_file nvsmi_numreqsample0_iter100_max1000.out
# /app/tensorrt_llm/examples/llama
python ../benchmark_trtllm.py --tokenizer_dir ./meta-llama/Llama-2-7b-chat-hf_tokenizer/ --engine_dir ./llama/7B/trt_engines/fp16/1-gpu-2-batch/ --dataset_path ../ShareGPT_V3_unfiltered_cleaned_split.json --num_requests_sample 0 --max_batch_size 2 --max_input_tokens 1000 --max_output_tokens 1000 --output_dir ./outputs/llama/7B/fp16/1-gpu-2-batch --output_file bmark_numreqsample0_iter100_max1000.out --random_seed 42 --num_iterations 100 --use_prompt_formatting --add_special_tokens
# /dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm
sudo docker cp c71f5b2d5461:/app/tensorrt_llm/examples/llama/outputs/llama/7B/fp16/1-gpu-2-batch/bmark_numreqsample0_iter100_max1000.out outputs/llama/7B/fp16/1-gpu-2-batch/bmark_numreqsample0_iter100_max1000.out

# 1 gpu 2 batch 500 max
# /dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm
python benchmarking/nvsmi_monitor.py --output_dir outputs/llama/7B/fp16/1-gpu-2-batch --output_file nvsmi_numreqsample0_iter100_max500.out
# /app/tensorrt_llm/examples/llama
python ../benchmark_trtllm.py --tokenizer_dir ./meta-llama/Llama-2-7b-chat-hf_tokenizer/ --engine_dir ./llama/7B/trt_engines/fp16/1-gpu-2-batch/ --dataset_path ../ShareGPT_V3_unfiltered_cleaned_split.json --num_requests_sample 0 --max_batch_size 2 --max_input_tokens 500 --max_output_tokens 500 --output_dir ./outputs/llama/7B/fp16/1-gpu-2-batch --output_file bmark_numreqsample0_iter100_max500.out --random_seed 42 --num_iterations 100 --use_prompt_formatting --add_special_tokens
# /dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm
sudo docker cp c71f5b2d5461:/app/tensorrt_llm/examples/llama/outputs/llama/7B/fp16/1-gpu-2-batch/bmark_numreqsample0_iter100_max500.out outputs/llama/7B/fp16/1-gpu-2-batch/bmark_numreqsample0_iter100_max500.out

# 1 gpu 4 batch 1000 max
# /dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm
python benchmarking/nvsmi_monitor.py --output_dir outputs/llama/7B/fp16/1-gpu-4-batch --output_file nvsmi_numreqsample0_iter100_max1000.out
# /app/tensorrt_llm/examples/llama
python ../benchmark_trtllm.py --tokenizer_dir ./meta-llama/Llama-2-7b-chat-hf_tokenizer/ --engine_dir ./llama/7B/trt_engines/fp16/1-gpu-4-batch/ --dataset_path ../ShareGPT_V3_unfiltered_cleaned_split.json --num_requests_sample 0 --max_batch_size 4 --max_input_tokens 1000 --max_output_tokens 1000 --output_dir ./outputs/llama/7B/fp16/1-gpu-4-batch --output_file bmark_numreqsample0_iter100_max1000.out --random_seed 42 --num_iterations 100 --use_prompt_formatting --add_special_tokens
# /dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm
sudo docker cp c71f5b2d5461:/app/tensorrt_llm/examples/llama/outputs/llama/7B/fp16/1-gpu-4-batch/bmark_numreqsample0_iter100_max1000.out outputs/llama/7B/fp16/1-gpu-4-batch/bmark_numreqsample0_iter100_max1000.out

# 1 gpu 4 batch 500 max
# /dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm
python benchmarking/nvsmi_monitor.py --output_dir outputs/llama/7B/fp16/1-gpu-4-batch --output_file nvsmi_numreqsample0_iter100_max500.out
# /app/tensorrt_llm/examples/llama
python ../benchmark_trtllm.py --tokenizer_dir ./meta-llama/Llama-2-7b-chat-hf_tokenizer/ --engine_dir ./llama/7B/trt_engines/fp16/1-gpu-4-batch/ --dataset_path ../ShareGPT_V3_unfiltered_cleaned_split.json --num_requests_sample 0 --max_batch_size 4 --max_input_tokens 500 --max_output_tokens 500 --output_dir ./outputs/llama/7B/fp16/1-gpu-4-batch --output_file bmark_numreqsample0_iter100_max500.out --random_seed 42 --num_iterations 100 --use_prompt_formatting --add_special_tokens
# /dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm
sudo docker cp c71f5b2d5461:/app/tensorrt_llm/examples/llama/outputs/llama/7B/fp16/1-gpu-4-batch/bmark_numreqsample0_iter100_max500.out outputs/llama/7B/fp16/1-gpu-4-batch/bmark_numreqsample0_iter100_max500.out

# TODO: OOM
# 1 gpu 6 batch 1000 max
# /dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm
python benchmarking/nvsmi_monitor.py --output_dir outputs/llama/7B/fp16/1-gpu-6-batch --output_file nvsmi_numreqsample0_iter100_max1000.out
# /app/tensorrt_llm/examples/llama
python ../benchmark_trtllm.py --tokenizer_dir ./meta-llama/Llama-2-7b-chat-hf_tokenizer/ --engine_dir ./llama/7B/trt_engines/fp16/1-gpu-6-batch/ --dataset_path ../ShareGPT_V3_unfiltered_cleaned_split.json --num_requests_sample 0 --max_batch_size 6 --max_input_tokens 1000 --max_output_tokens 1000 --output_dir ./outputs/llama/7B/fp16/1-gpu-6-batch --output_file bmark_numreqsample0_iter100_max1000.out --random_seed 42 --num_iterations 100 --use_prompt_formatting --add_special_tokens
# /dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm
sudo docker cp c71f5b2d5461:/app/tensorrt_llm/examples/llama/outputs/llama/7B/fp16/1-gpu-6-batch/bmark_numreqsample0_iter100_max1000.out outputs/llama/7B/fp16/1-gpu-6-batch/bmark_numreqsample0_iter100_max1000.out

# 1 gpu 6 batch 500 max
# /dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm
python benchmarking/nvsmi_monitor.py --output_dir outputs/llama/7B/fp16/1-gpu-6-batch --output_file nvsmi_numreqsample0_iter100_max500.out
# /app/tensorrt_llm/examples/llama
python ../benchmark_trtllm.py --tokenizer_dir ./meta-llama/Llama-2-7b-chat-hf_tokenizer/ --engine_dir ./llama/7B/trt_engines/fp16/1-gpu-6-batch/ --dataset_path ../ShareGPT_V3_unfiltered_cleaned_split.json --num_requests_sample 0 --max_batch_size 6 --max_input_tokens 500 --max_output_tokens 500 --output_dir ./outputs/llama/7B/fp16/1-gpu-6-batch --output_file bmark_numreqsample0_iter100_max500.out --random_seed 42 --num_iterations 100 --use_prompt_formatting --add_special_tokens
# /dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm
sudo docker cp c71f5b2d5461:/app/tensorrt_llm/examples/llama/outputs/llama/7B/fp16/1-gpu-6-batch/bmark_numreqsample0_iter100_max500.out outputs/llama/7B/fp16/1-gpu-6-batch/bmark_numreqsample0_iter100_max500.out

# 1 gpu 8 batch 1000 max
# /dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm
python benchmarking/nvsmi_monitor.py --output_dir outputs/llama/7B/fp16/1-gpu-8-batch --output_file nvsmi_numreqsample0_iter100_max1000.out
# /app/tensorrt_llm/examples/llama
python ../benchmark_trtllm.py --tokenizer_dir ./meta-llama/Llama-2-7b-chat-hf_tokenizer/ --engine_dir ./llama/7B/trt_engines/fp16/1-gpu-8-batch/ --dataset_path ../ShareGPT_V3_unfiltered_cleaned_split.json --num_requests_sample 0 --max_batch_size 8 --max_input_tokens 1000 --max_output_tokens 1000 --output_dir ./outputs/llama/7B/fp16/1-gpu-8-batch --output_file bmark_numreqsample0_iter100_max1000.out --random_seed 42 --num_iterations 100 --use_prompt_formatting --add_special_tokens
# /dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm
sudo docker cp c71f5b2d5461:/app/tensorrt_llm/examples/llama/outputs/llama/7B/fp16/1-gpu-8-batch/bmark_numreqsample0_iter100_max1000.out outputs/llama/7B/fp16/1-gpu-8-batch/bmark_numreqsample0_iter100_max1000.out

# 1 gpu 8 batch 500 max
# /dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm
python benchmarking/nvsmi_monitor.py --output_dir outputs/llama/7B/fp16/1-gpu-8-batch --output_file nvsmi_numreqsample0_iter100_max500.out
# /app/tensorrt_llm/examples/llama
python ../benchmark_trtllm.py --tokenizer_dir ./meta-llama/Llama-2-7b-chat-hf_tokenizer/ --engine_dir ./llama/7B/trt_engines/fp16/1-gpu-8-batch/ --dataset_path ../ShareGPT_V3_unfiltered_cleaned_split.json --num_requests_sample 0 --max_batch_size 8 --max_input_tokens 500 --max_output_tokens 500 --output_dir ./outputs/llama/7B/fp16/1-gpu-8-batch --output_file bmark_numreqsample0_iter100_max500.out --random_seed 42 --num_iterations 100 --use_prompt_formatting --add_special_tokens
# /dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm
sudo docker cp c71f5b2d5461:/app/tensorrt_llm/examples/llama/outputs/llama/7B/fp16/1-gpu-8-batch/bmark_numreqsample0_iter100_max500.out outputs/llama/7B/fp16/1-gpu-8-batch/bmark_numreqsample0_iter100_max500.out

# 1 gpu 10 batch 1000 max
# /dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm
python benchmarking/nvsmi_monitor.py --output_dir outputs/llama/7B/fp16/1-gpu-10-batch --output_file nvsmi_numreqsample0_iter100_max1000.out
# /app/tensorrt_llm/examples/llama
python ../benchmark_trtllm.py --tokenizer_dir ./meta-llama/Llama-2-7b-chat-hf_tokenizer/ --engine_dir ./llama/7B/trt_engines/fp16/1-gpu-10-batch/ --dataset_path ../ShareGPT_V3_unfiltered_cleaned_split.json --num_requests_sample 0 --max_batch_size 10 --max_input_tokens 1000 --max_output_tokens 1000 --output_dir ./outputs/llama/7B/fp16/1-gpu-10-batch --output_file bmark_numreqsample0_iter100_max1000.out --random_seed 42 --num_iterations 100 --use_prompt_formatting --add_special_tokens
# /dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm
sudo docker cp c71f5b2d5461:/app/tensorrt_llm/examples/llama/outputs/llama/7B/fp16/1-gpu-10-batch/bmark_numreqsample0_iter100_max1000.out outputs/llama/7B/fp16/1-gpu-10-batch/bmark_numreqsample0_iter100_max1000.out

# 1 gpu 10 batch 500 max
# /dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm
python benchmarking/nvsmi_monitor.py --output_dir outputs/llama/7B/fp16/1-gpu-10-batch --output_file nvsmi_numreqsample0_iter100_max500.out
# /app/tensorrt_llm/examples/llama
python ../benchmark_trtllm.py --tokenizer_dir ./meta-llama/Llama-2-7b-chat-hf_tokenizer/ --engine_dir ./llama/7B/trt_engines/fp16/1-gpu-10-batch/ --dataset_path ../ShareGPT_V3_unfiltered_cleaned_split.json --num_requests_sample 0 --max_batch_size 10 --max_input_tokens 500 --max_output_tokens 500 --output_dir ./outputs/llama/7B/fp16/1-gpu-10-batch --output_file bmark_numreqsample0_iter100_max500.out --random_seed 42 --num_iterations 100 --use_prompt_formatting --add_special_tokens
# /dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm
sudo docker cp c71f5b2d5461:/app/tensorrt_llm/examples/llama/outputs/llama/7B/fp16/1-gpu-10-batch/bmark_numreqsample0_iter100_max500.out outputs/llama/7B/fp16/1-gpu-10-batch/bmark_numreqsample0_iter100_max500.out

# 1 gpu 12 batch 1000 max
# /dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm
python benchmarking/nvsmi_monitor.py --output_dir outputs/llama/7B/fp16/1-gpu-12-batch --output_file nvsmi_numreqsample0_iter100_max1000.out
# /app/tensorrt_llm/examples/llama
python ../benchmark_trtllm.py --tokenizer_dir ./meta-llama/Llama-2-7b-chat-hf_tokenizer/ --engine_dir ./llama/7B/trt_engines/fp16/1-gpu-12-batch/ --dataset_path ../ShareGPT_V3_unfiltered_cleaned_split.json --num_requests_sample 0 --max_batch_size 12 --max_input_tokens 1000 --max_output_tokens 1000 --output_dir ./outputs/llama/7B/fp16/1-gpu-12-batch --output_file bmark_numreqsample0_iter100_max1000.out --random_seed 42 --num_iterations 100 --use_prompt_formatting --add_special_tokens
# /dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm
sudo docker cp c71f5b2d5461:/app/tensorrt_llm/examples/llama/outputs/llama/7B/fp16/1-gpu-12-batch/bmark_numreqsample0_iter100_max1000.out outputs/llama/7B/fp16/1-gpu-12-batch/bmark_numreqsample0_iter100_max1000.out

# 1 gpu 12 batch 500 max
# /dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm
python benchmarking/nvsmi_monitor.py --output_dir outputs/llama/7B/fp16/1-gpu-12-batch --output_file nvsmi_numreqsample0_iter100_max500.out
# /app/tensorrt_llm/examples/llama
python ../benchmark_trtllm.py --tokenizer_dir ./meta-llama/Llama-2-7b-chat-hf_tokenizer/ --engine_dir ./llama/7B/trt_engines/fp16/1-gpu-12-batch/ --dataset_path ../ShareGPT_V3_unfiltered_cleaned_split.json --num_requests_sample 0 --max_batch_size 12 --max_input_tokens 500 --max_output_tokens 500 --output_dir ./outputs/llama/7B/fp16/1-gpu-12-batch --output_file bmark_numreqsample0_iter100_max500.out --random_seed 42 --num_iterations 100 --use_prompt_formatting --add_special_tokens
# /dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm
sudo docker cp c71f5b2d5461:/app/tensorrt_llm/examples/llama/outputs/llama/7B/fp16/1-gpu-12-batch/bmark_numreqsample0_iter100_max500.out outputs/llama/7B/fp16/1-gpu-12-batch/bmark_numreqsample0_iter100_max500.out

# 1 gpu 14 batch 1000 max
# /dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm
python benchmarking/nvsmi_monitor.py --output_dir outputs/llama/7B/fp16/1-gpu-14-batch --output_file nvsmi_numreqsample0_iter100_max1000.out
# /app/tensorrt_llm/examples/llama
python ../benchmark_trtllm.py --tokenizer_dir ./meta-llama/Llama-2-7b-chat-hf_tokenizer/ --engine_dir ./llama/7B/trt_engines/fp16/1-gpu-14-batch/ --dataset_path ../ShareGPT_V3_unfiltered_cleaned_split.json --num_requests_sample 0 --max_batch_size 14 --max_input_tokens 1000 --max_output_tokens 1000 --output_dir ./outputs/llama/7B/fp16/1-gpu-14-batch --output_file bmark_numreqsample0_iter100_max1000.out --random_seed 42 --num_iterations 100 --use_prompt_formatting --add_special_tokens
# /dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm
sudo docker cp c71f5b2d5461:/app/tensorrt_llm/examples/llama/outputs/llama/7B/fp16/1-gpu-14-batch/bmark_numreqsample0_iter100_max1000.out outputs/llama/7B/fp16/1-gpu-14-batch/bmark_numreqsample0_iter100_max1000.out

# 1 gpu 14 batch 500 max
# /dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm
python benchmarking/nvsmi_monitor.py --output_dir outputs/llama/7B/fp16/1-gpu-14-batch --output_file nvsmi_numreqsample0_iter100_max500.out
# /app/tensorrt_llm/examples/llama
python ../benchmark_trtllm.py --tokenizer_dir ./meta-llama/Llama-2-7b-chat-hf_tokenizer/ --engine_dir ./llama/7B/trt_engines/fp16/1-gpu-14-batch/ --dataset_path ../ShareGPT_V3_unfiltered_cleaned_split.json --num_requests_sample 0 --max_batch_size 14 --max_input_tokens 500 --max_output_tokens 500 --output_dir ./outputs/llama/7B/fp16/1-gpu-14-batch --output_file bmark_numreqsample0_iter100_max500.out --random_seed 42 --num_iterations 100 --use_prompt_formatting --add_special_tokens
# /dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm
sudo docker cp c71f5b2d5461:/app/tensorrt_llm/examples/llama/outputs/llama/7B/fp16/1-gpu-14-batch/bmark_numreqsample0_iter100_max500.out outputs/llama/7B/fp16/1-gpu-14-batch/bmark_numreqsample0_iter100_max500.out

# TODO: OOM
# 1 gpu 16 batch 1000 max
# /dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm
python benchmarking/nvsmi_monitor.py --output_dir outputs/llama/7B/fp16/1-gpu-16-batch --output_file nvsmi_numreqsample0_iter100_max1000.out
# /app/tensorrt_llm/examples/llama
python ../benchmark_trtllm.py --tokenizer_dir ./meta-llama/Llama-2-7b-chat-hf_tokenizer/ --engine_dir ./llama/7B/trt_engines/fp16/1-gpu-16-batch/ --dataset_path ../ShareGPT_V3_unfiltered_cleaned_split.json --num_requests_sample 0 --max_batch_size 16 --max_input_tokens 1000 --max_output_tokens 1000 --output_dir ./outputs/llama/7B/fp16/1-gpu-16-batch --output_file bmark_numreqsample0_iter100_max1000.out --random_seed 42 --num_iterations 100 --use_prompt_formatting --add_special_tokens
# /dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm
sudo docker cp c71f5b2d5461:/app/tensorrt_llm/examples/llama/outputs/llama/7B/fp16/1-gpu-16-batch/bmark_numreqsample0_iter100_max1000.out outputs/llama/7B/fp16/1-gpu-16-batch/bmark_numreqsample0_iter100_max1000.out

# 1 gpu 16 batch 500 max
# /dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm
python benchmarking/nvsmi_monitor.py --output_dir outputs/llama/7B/fp16/1-gpu-16-batch --output_file nvsmi_numreqsample0_iter100_max500.out
# /app/tensorrt_llm/examples/llama
python ../benchmark_trtllm.py --tokenizer_dir ./meta-llama/Llama-2-7b-chat-hf_tokenizer/ --engine_dir ./llama/7B/trt_engines/fp16/1-gpu-16-batch/ --dataset_path ../ShareGPT_V3_unfiltered_cleaned_split.json --num_requests_sample 0 --max_batch_size 16 --max_input_tokens 500 --max_output_tokens 500 --output_dir ./outputs/llama/7B/fp16/1-gpu-16-batch --output_file bmark_numreqsample0_iter100_max500.out --random_seed 42 --num_iterations 100 --use_prompt_formatting --add_special_tokens
# /dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm
sudo docker cp c71f5b2d5461:/app/tensorrt_llm/examples/llama/outputs/llama/7B/fp16/1-gpu-16-batch/bmark_numreqsample0_iter100_max500.out outputs/llama/7B/fp16/1-gpu-16-batch/bmark_numreqsample0_iter100_max500.out

# 1 gpu 18 batch 500 max
# /dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm
python benchmarking/nvsmi_monitor.py --output_dir outputs/llama/7B/fp16/1-gpu-18-batch --output_file nvsmi_numreqsample0_iter100_max500.out
# /app/tensorrt_llm/examples/llama
python ../benchmark_trtllm.py --tokenizer_dir ./meta-llama/Llama-2-7b-chat-hf_tokenizer/ --engine_dir ./llama/7B/trt_engines/fp16/1-gpu-18-batch/ --dataset_path ../ShareGPT_V3_unfiltered_cleaned_split.json --num_requests_sample 0 --max_batch_size 18 --max_input_tokens 500 --max_output_tokens 500 --output_dir ./outputs/llama/7B/fp16/1-gpu-18-batch --output_file bmark_numreqsample0_iter100_max500.out --random_seed 42 --num_iterations 100 --use_prompt_formatting --add_special_tokens
# /dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm
sudo docker cp c71f5b2d5461:/app/tensorrt_llm/examples/llama/outputs/llama/7B/fp16/1-gpu-18-batch/bmark_numreqsample0_iter100_max500.out outputs/llama/7B/fp16/1-gpu-18-batch/bmark_numreqsample0_iter100_max500.out

# 1 gpu 20 batch 500 max
# /dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm
python benchmarking/nvsmi_monitor.py --output_dir outputs/llama/7B/fp16/1-gpu-20-batch --output_file nvsmi_numreqsample0_iter100_max500.out
# /app/tensorrt_llm/examples/llama
python ../benchmark_trtllm.py --tokenizer_dir ./meta-llama/Llama-2-7b-chat-hf_tokenizer/ --engine_dir ./llama/7B/trt_engines/fp16/1-gpu-20-batch/ --dataset_path ../ShareGPT_V3_unfiltered_cleaned_split.json --num_requests_sample 0 --max_batch_size 20 --max_input_tokens 500 --max_output_tokens 500 --output_dir ./outputs/llama/7B/fp16/1-gpu-20-batch --output_file bmark_numreqsample0_iter100_max500.out --random_seed 42 --num_iterations 100 --use_prompt_formatting --add_special_tokens
# /dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm
sudo docker cp c71f5b2d5461:/app/tensorrt_llm/examples/llama/outputs/llama/7B/fp16/1-gpu-20-batch/bmark_numreqsample0_iter100_max500.out outputs/llama/7B/fp16/1-gpu-20-batch/bmark_numreqsample0_iter100_max500.out

# 1 gpu 22 batch 500 max
# /dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm
python benchmarking/nvsmi_monitor.py --output_dir outputs/llama/7B/fp16/1-gpu-22-batch --output_file nvsmi_numreqsample0_iter100_max500.out
# /app/tensorrt_llm/examples/llama
python ../benchmark_trtllm.py --tokenizer_dir ./meta-llama/Llama-2-7b-chat-hf_tokenizer/ --engine_dir ./llama/7B/trt_engines/fp16/1-gpu-22-batch/ --dataset_path ../ShareGPT_V3_unfiltered_cleaned_split.json --num_requests_sample 0 --max_batch_size 22 --max_input_tokens 500 --max_output_tokens 500 --output_dir ./outputs/llama/7B/fp16/1-gpu-22-batch --output_file bmark_numreqsample0_iter100_max500.out --random_seed 42 --num_iterations 100 --use_prompt_formatting --add_special_tokens
# /dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm
sudo docker cp c71f5b2d5461:/app/tensorrt_llm/examples/llama/outputs/llama/7B/fp16/1-gpu-22-batch/bmark_numreqsample0_iter100_max500.out outputs/llama/7B/fp16/1-gpu-22-batch/bmark_numreqsample0_iter100_max500.out

# 1 gpu 24 batch 500 max
# /dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm
python benchmarking/nvsmi_monitor.py --output_dir outputs/llama/7B/fp16/1-gpu-24-batch --output_file nvsmi_numreqsample0_iter100_max500.out
# /app/tensorrt_llm/examples/llama
python ../benchmark_trtllm.py --tokenizer_dir ./meta-llama/Llama-2-7b-chat-hf_tokenizer/ --engine_dir ./llama/7B/trt_engines/fp16/1-gpu-24-batch/ --dataset_path ../ShareGPT_V3_unfiltered_cleaned_split.json --num_requests_sample 0 --max_batch_size 24 --max_input_tokens 500 --max_output_tokens 500 --output_dir ./outputs/llama/7B/fp16/1-gpu-24-batch --output_file bmark_numreqsample0_iter100_max500.out --random_seed 42 --num_iterations 100 --use_prompt_formatting --add_special_tokens
# /dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm
sudo docker cp c71f5b2d5461:/app/tensorrt_llm/examples/llama/outputs/llama/7B/fp16/1-gpu-24-batch/bmark_numreqsample0_iter100_max500.out outputs/llama/7B/fp16/1-gpu-24-batch/bmark_numreqsample0_iter100_max500.out

# 1 gpu 26 batch 500 max
# /dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm
python benchmarking/nvsmi_monitor.py --output_dir outputs/llama/7B/fp16/1-gpu-26-batch --output_file nvsmi_numreqsample0_iter100_max500.out
# /app/tensorrt_llm/examples/llama
python ../benchmark_trtllm.py --tokenizer_dir ./meta-llama/Llama-2-7b-chat-hf_tokenizer/ --engine_dir ./llama/7B/trt_engines/fp16/1-gpu-26-batch/ --dataset_path ../ShareGPT_V3_unfiltered_cleaned_split.json --num_requests_sample 0 --max_batch_size 26 --max_input_tokens 500 --max_output_tokens 500 --output_dir ./outputs/llama/7B/fp16/1-gpu-26-batch --output_file bmark_numreqsample0_iter100_max500.out --random_seed 42 --num_iterations 100 --use_prompt_formatting --add_special_tokens
# /dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm
sudo docker cp c71f5b2d5461:/app/tensorrt_llm/examples/llama/outputs/llama/7B/fp16/1-gpu-26-batch/bmark_numreqsample0_iter100_max500.out outputs/llama/7B/fp16/1-gpu-26-batch/bmark_numreqsample0_iter100_max500.out

# TODO: UPDATED BENCHMARK SCRIPTS START HERE
# TODO: PREVIOUS RESULTS SHOULD BE REFORMATTED TO MATCH THE SAME FORMAT AS THE NEW NVSMI LOGGER
# TESTING
python benchmarking/nvsmi_monitor.py --output_dir outputs/llama/7B/fp16/1-gpu-1-batch --output_file nvsmi_testing.out --container_id c71f5b2d5461 --container_output_dir /app/tensorrt_llm/examples/llama --container_stop_file container_stop.txt
python ../benchmark_trtllm.py --tokenizer_dir ./meta-llama/Llama-2-7b-chat-hf_tokenizer/ --engine_dir ./llama/7B/trt_engines/fp16/1-gpu-1-batch/ --dataset_path ../ShareGPT_V3_unfiltered_cleaned_split.json --num_requests_sample 0 --max_batch_size 1 --max_input_tokens 1000 --max_output_tokens 1000 --output_dir /app/tensorrt_llm/examples/llama/outputs/llama/7B/fp16/1-gpu-1-batch --output_file bmark_testing.out --container_output_dir /app/tensorrt_llm/examples/llama --container_stop_file container_stop.txt --random_seed 42 --num_iterations 10 --use_prompt_formatting --add_special_tokens

# 1 gpu 28 batch 500 max
# /dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm
python benchmarking/nvsmi_monitor.py --output_dir outputs/llama/7B/fp16/1-gpu-28-batch --output_file nvsmi_numreqsample0_iter100_max500_v10032gb.out --container_id c71f5b2d5461 --container_output_dir /app/tensorrt_llm/examples/llama --container_stop_file container_stop.txt
# /app/tensorrt_llm/examples/llama
python ../benchmark_trtllm.py --tokenizer_dir ./meta-llama/Llama-2-7b-chat-hf_tokenizer/ --engine_dir ./llama/7B/trt_engines/fp16/1-gpu-28-batch/ --dataset_path ../ShareGPT_V3_unfiltered_cleaned_split.json --num_requests_sample 0 --max_batch_size 28 --max_input_tokens 500 --max_output_tokens 500 --output_dir /app/tensorrt_llm/examples/llama/outputs/llama/7B/fp16/1-gpu-28-batch --output_file bmark_numreqsample0_iter100_max500_v10032gb.out --container_output_dir /app/tensorrt_llm/examples/llama --container_stop_file container_stop.txt --random_seed 42 --num_iterations 100 --use_prompt_formatting --add_special_tokens
# /dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm
sudo docker cp c71f5b2d5461:/app/tensorrt_llm/examples/llama/outputs/llama/7B/fp16/1-gpu-28-batch/bmark_numreqsample0_iter100_max500_v10032gb.out outputs/llama/7B/fp16/1-gpu-28-batch/bmark_numreqsample0_iter100_max500_v10032gb.out

# 1 gpu 30 batch 500 max
# /dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm
python benchmarking/nvsmi_monitor.py --output_dir outputs/llama/7B/fp16/1-gpu-30-batch --output_file nvsmi_numreqsample0_iter100_max500_v10032gb.out --container_id c71f5b2d5461 --container_output_dir /app/tensorrt_llm/examples/llama --container_stop_file container_stop.txt
# /app/tensorrt_llm/examples/llama
python ../benchmark_trtllm.py --tokenizer_dir ./meta-llama/Llama-2-7b-chat-hf_tokenizer/ --engine_dir ./llama/7B/trt_engines/fp16/1-gpu-30-batch/ --dataset_path ../ShareGPT_V3_unfiltered_cleaned_split.json --num_requests_sample 0 --max_batch_size 30 --max_input_tokens 500 --max_output_tokens 500 --output_dir /app/tensorrt_llm/examples/llama/outputs/llama/7B/fp16/1-gpu-30-batch --output_file bmark_numreqsample0_iter100_max500_v10032gb.out --container_output_dir /app/tensorrt_llm/examples/llama --container_stop_file container_stop.txt --random_seed 42 --num_iterations 100 --use_prompt_formatting --add_special_tokens
# /dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm
sudo docker cp c71f5b2d5461:/app/tensorrt_llm/examples/llama/outputs/llama/7B/fp16/1-gpu-30-batch/bmark_numreqsample0_iter100_max500_v10032gb.out outputs/llama/7B/fp16/1-gpu-30-batch/bmark_numreqsample0_iter100_max500_v10032gb.out

# 1 gpu 32 batch 500 max
# /dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm
python benchmarking/nvsmi_monitor.py --output_dir outputs/llama/7B/fp16/1-gpu-32-batch --output_file nvsmi_numreqsample0_iter100_max500_v10032gb.out --container_id c71f5b2d5461 --container_output_dir /app/tensorrt_llm/examples/llama --container_stop_file container_stop.txt
# /app/tensorrt_llm/examples/llama
python ../benchmark_trtllm.py --tokenizer_dir ./meta-llama/Llama-2-7b-chat-hf_tokenizer/ --engine_dir ./llama/7B/trt_engines/fp16/1-gpu-32-batch/ --dataset_path ../ShareGPT_V3_unfiltered_cleaned_split.json --num_requests_sample 0 --max_batch_size 32 --max_input_tokens 500 --max_output_tokens 500 --output_dir /app/tensorrt_llm/examples/llama/outputs/llama/7B/fp16/1-gpu-32-batch --output_file bmark_numreqsample0_iter100_max500_v10032gb.out --container_output_dir /app/tensorrt_llm/examples/llama --container_stop_file container_stop.txt --random_seed 42 --num_iterations 100 --use_prompt_formatting --add_special_tokens
# /dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm
sudo docker cp c71f5b2d5461:/app/tensorrt_llm/examples/llama/outputs/llama/7B/fp16/1-gpu-32-batch/bmark_numreqsample0_iter100_max500_v10032gb.out outputs/llama/7B/fp16/1-gpu-32-batch/bmark_numreqsample0_iter100_max500_v10032gb.out




# 4-BIT WEIGHT-QUANTIZED EXPERIMENTS
# 1 gpu 1 batch 1000 max
# /dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm
python benchmarking/nvsmi_monitor.py --output_dir outputs/llama/7B/fp16_wq4/1-gpu-1-batch --output_file nvsmi_numreqsample0_iter100_max1000_v10032gb.out --container_id c71f5b2d5461 --container_output_dir /app/tensorrt_llm/examples/llama --container_stop_file container_stop.txt --gpu_type v10032gb
# /app/tensorrt_llm/examples/llama
python ../benchmark_trtllm.py --tokenizer_dir ./meta-llama/Llama-2-7b-chat-hf_tokenizer/ --engine_dir ./llama/7B/trt_engines/fp16_wq4/1-gpu-1-batch/ --dataset_path ../ShareGPT_V3_unfiltered_cleaned_split.json --num_requests_sample 0 --max_batch_size 1 --max_input_tokens 1000 --max_output_tokens 1000 --output_dir /app/tensorrt_llm/examples/llama/outputs/llama/7B/fp16_wq4/1-gpu-1-batch --output_file bmark_numreqsample0_iter100_max1000_v10032gb.out --container_output_dir /app/tensorrt_llm/examples/llama --container_stop_file container_stop.txt --random_seed 42 --num_iterations 100 --use_prompt_formatting --add_special_tokens
# /dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm
sudo docker cp c71f5b2d5461:/app/tensorrt_llm/examples/llama/outputs/llama/7B/fp16_wq4/1-gpu-1-batch/bmark_numreqsample0_iter100_max1000_v10032gb.out outputs/llama/7B/fp16_wq4/1-gpu-1-batch/bmark_numreqsample0_iter100_max1000_v10032gb.out

# 1 gpu 4 batch 1000 max
# /dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm
python benchmarking/nvsmi_monitor.py --output_dir outputs/llama/7B/fp16_wq4/1-gpu-4-batch --output_file nvsmi_numreqsample0_iter100_max1000_v10032gb.out --container_id c71f5b2d5461 --container_output_dir /app/tensorrt_llm/examples/llama --container_stop_file container_stop.txt --gpu_type v10032gb
# /app/tensorrt_llm/examples/llama
python ../benchmark_trtllm.py --tokenizer_dir ./meta-llama/Llama-2-7b-chat-hf_tokenizer/ --engine_dir ./llama/7B/trt_engines/fp16_wq4/1-gpu-4-batch/ --dataset_path ../ShareGPT_V3_unfiltered_cleaned_split.json --num_requests_sample 0 --max_batch_size 4 --max_input_tokens 1000 --max_output_tokens 1000 --output_dir /app/tensorrt_llm/examples/llama/outputs/llama/7B/fp16_wq4/1-gpu-4-batch --output_file bmark_numreqsample0_iter100_max1000_v10032gb.out --container_output_dir /app/tensorrt_llm/examples/llama --container_stop_file container_stop.txt --random_seed 42 --num_iterations 100 --use_prompt_formatting --add_special_tokens
# /dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm
sudo docker cp c71f5b2d5461:/app/tensorrt_llm/examples/llama/outputs/llama/7B/fp16_wq4/1-gpu-4-batch/bmark_numreqsample0_iter100_max1000_v10032gb.out outputs/llama/7B/fp16_wq4/1-gpu-4-batch/bmark_numreqsample0_iter100_max1000_v10032gb.out

# 1 gpu 8 batch 1000 max
# /dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm
python benchmarking/nvsmi_monitor.py --output_dir outputs/llama/7B/fp16_wq4/1-gpu-8-batch --output_file nvsmi_numreqsample0_iter100_max1000_v10032gb.out --container_id c71f5b2d5461 --container_output_dir /app/tensorrt_llm/examples/llama --container_stop_file container_stop.txt --gpu_type v10032gb
# /app/tensorrt_llm/examples/llama
python ../benchmark_trtllm.py --tokenizer_dir ./meta-llama/Llama-2-7b-chat-hf_tokenizer/ --engine_dir ./llama/7B/trt_engines/fp16_wq4/1-gpu-8-batch/ --dataset_path ../ShareGPT_V3_unfiltered_cleaned_split.json --num_requests_sample 0 --max_batch_size 8 --max_input_tokens 1000 --max_output_tokens 1000 --output_dir /app/tensorrt_llm/examples/llama/outputs/llama/7B/fp16_wq4/1-gpu-8-batch --output_file bmark_numreqsample0_iter100_max1000_v10032gb.out --container_output_dir /app/tensorrt_llm/examples/llama --container_stop_file container_stop.txt --random_seed 42 --num_iterations 100 --use_prompt_formatting --add_special_tokens
# /dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm
sudo docker cp c71f5b2d5461:/app/tensorrt_llm/examples/llama/outputs/llama/7B/fp16_wq4/1-gpu-8-batch/bmark_numreqsample0_iter100_max1000_v10032gb.out outputs/llama/7B/fp16_wq4/1-gpu-8-batch/bmark_numreqsample0_iter100_max1000_v10032gb.out

# 1 gpu 12 batch 1000 max
# /dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm
python benchmarking/nvsmi_monitor.py --output_dir outputs/llama/7B/fp16_wq4/1-gpu-12-batch --output_file nvsmi_numreqsample0_iter100_max1000_v10032gb.out --container_id c71f5b2d5461 --container_output_dir /app/tensorrt_llm/examples/llama --container_stop_file container_stop.txt --gpu_type v10032gb
# /app/tensorrt_llm/examples/llama
python ../benchmark_trtllm.py --tokenizer_dir ./meta-llama/Llama-2-7b-chat-hf_tokenizer/ --engine_dir ./llama/7B/trt_engines/fp16_wq4/1-gpu-12-batch/ --dataset_path ../ShareGPT_V3_unfiltered_cleaned_split.json --num_requests_sample 0 --max_batch_size 12 --max_input_tokens 1000 --max_output_tokens 1000 --output_dir /app/tensorrt_llm/examples/llama/outputs/llama/7B/fp16_wq4/1-gpu-12-batch --output_file bmark_numreqsample0_iter100_max1000_v10032gb.out --container_output_dir /app/tensorrt_llm/examples/llama --container_stop_file container_stop.txt --random_seed 42 --num_iterations 100 --use_prompt_formatting --add_special_tokens
# /dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm
sudo docker cp c71f5b2d5461:/app/tensorrt_llm/examples/llama/outputs/llama/7B/fp16_wq4/1-gpu-12-batch/bmark_numreqsample0_iter100_max1000_v10032gb.out outputs/llama/7B/fp16_wq4/1-gpu-12-batch/bmark_numreqsample0_iter100_max1000_v10032gb.out

# 1 gpu 16 batch 1000 max
# /dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm
python benchmarking/nvsmi_monitor.py --output_dir outputs/llama/7B/fp16_wq4/1-gpu-16-batch --output_file nvsmi_numreqsample0_iter100_max1000_v10032gb.out --container_id c71f5b2d5461 --container_output_dir /app/tensorrt_llm/examples/llama --container_stop_file container_stop.txt --gpu_type v10032gb
# /app/tensorrt_llm/examples/llama
python ../benchmark_trtllm.py --tokenizer_dir ./meta-llama/Llama-2-7b-chat-hf_tokenizer/ --engine_dir ./llama/7B/trt_engines/fp16_wq4/1-gpu-16-batch/ --dataset_path ../ShareGPT_V3_unfiltered_cleaned_split.json --num_requests_sample 0 --max_batch_size 16 --max_input_tokens 1000 --max_output_tokens 1000 --output_dir /app/tensorrt_llm/examples/llama/outputs/llama/7B/fp16_wq4/1-gpu-16-batch --output_file bmark_numreqsample0_iter100_max1000_v10032gb.out --container_output_dir /app/tensorrt_llm/examples/llama --container_stop_file container_stop.txt --random_seed 42 --num_iterations 100 --use_prompt_formatting --add_special_tokens
# /dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm
sudo docker cp c71f5b2d5461:/app/tensorrt_llm/examples/llama/outputs/llama/7B/fp16_wq4/1-gpu-16-batch/bmark_numreqsample0_iter100_max1000_v10032gb.out outputs/llama/7B/fp16_wq4/1-gpu-16-batch/bmark_numreqsample0_iter100_max1000_v10032gb.out

# 1 gpu 20 batch 1000 max
# /dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm
python benchmarking/nvsmi_monitor.py --output_dir outputs/llama/7B/fp16_wq4/1-gpu-20-batch --output_file nvsmi_numreqsample0_iter100_max1000_v10032gb.out --container_id c71f5b2d5461 --container_output_dir /app/tensorrt_llm/examples/llama --container_stop_file container_stop.txt --gpu_type v10032gb
# /app/tensorrt_llm/examples/llama
python ../benchmark_trtllm.py --tokenizer_dir ./meta-llama/Llama-2-7b-chat-hf_tokenizer/ --engine_dir ./llama/7B/trt_engines/fp16_wq4/1-gpu-20-batch/ --dataset_path ../ShareGPT_V3_unfiltered_cleaned_split.json --num_requests_sample 0 --max_batch_size 20 --max_input_tokens 1000 --max_output_tokens 1000 --output_dir /app/tensorrt_llm/examples/llama/outputs/llama/7B/fp16_wq4/1-gpu-20-batch --output_file bmark_numreqsample0_iter100_max1000_v10032gb.out --container_output_dir /app/tensorrt_llm/examples/llama --container_stop_file container_stop.txt --random_seed 42 --num_iterations 100 --use_prompt_formatting --add_special_tokens
# /dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm
sudo docker cp c71f5b2d5461:/app/tensorrt_llm/examples/llama/outputs/llama/7B/fp16_wq4/1-gpu-20-batch/bmark_numreqsample0_iter100_max1000_v10032gb.out outputs/llama/7B/fp16_wq4/1-gpu-20-batch/bmark_numreqsample0_iter100_max1000_v10032gb.out

# 1 gpu 22 batch 1000 max
# /dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm
python benchmarking/nvsmi_monitor.py --output_dir outputs/llama/7B/fp16_wq4/1-gpu-22-batch --output_file nvsmi_numreqsample0_iter100_max1000_v10032gb.out --container_id c71f5b2d5461 --container_output_dir /app/tensorrt_llm/examples/llama --container_stop_file container_stop.txt --gpu_type v10032gb
# /app/tensorrt_llm/examples/llama
python ../benchmark_trtllm.py --tokenizer_dir ./meta-llama/Llama-2-7b-chat-hf_tokenizer/ --engine_dir ./llama/7B/trt_engines/fp16_wq4/1-gpu-22-batch/ --dataset_path ../ShareGPT_V3_unfiltered_cleaned_split.json --num_requests_sample 0 --max_batch_size 22 --max_input_tokens 1000 --max_output_tokens 1000 --output_dir /app/tensorrt_llm/examples/llama/outputs/llama/7B/fp16_wq4/1-gpu-22-batch --output_file bmark_numreqsample0_iter100_max1000_v10032gb.out --container_output_dir /app/tensorrt_llm/examples/llama --container_stop_file container_stop.txt --random_seed 42 --num_iterations 100 --use_prompt_formatting --add_special_tokens
# /dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm
sudo docker cp c71f5b2d5461:/app/tensorrt_llm/examples/llama/outputs/llama/7B/fp16_wq4/1-gpu-22-batch/bmark_numreqsample0_iter100_max1000_v10032gb.out outputs/llama/7B/fp16_wq4/1-gpu-22-batch/bmark_numreqsample0_iter100_max1000_v10032gb.out

# TODO: OOM
# 1 gpu 24 batch 1000 max
# /dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm
python benchmarking/nvsmi_monitor.py --output_dir outputs/llama/7B/fp16_wq4/1-gpu-24-batch --output_file nvsmi_numreqsample0_iter100_max1000_v10032gb.out --container_id c71f5b2d5461 --container_output_dir /app/tensorrt_llm/examples/llama --container_stop_file container_stop.txt --gpu_type v10032gb
# /app/tensorrt_llm/examples/llama
python ../benchmark_trtllm.py --tokenizer_dir ./meta-llama/Llama-2-7b-chat-hf_tokenizer/ --engine_dir ./llama/7B/trt_engines/fp16_wq4/1-gpu-24-batch/ --dataset_path ../ShareGPT_V3_unfiltered_cleaned_split.json --num_requests_sample 0 --max_batch_size 24 --max_input_tokens 1000 --max_output_tokens 1000 --output_dir /app/tensorrt_llm/examples/llama/outputs/llama/7B/fp16_wq4/1-gpu-24-batch --output_file bmark_numreqsample0_iter100_max1000_v10032gb.out --container_output_dir /app/tensorrt_llm/examples/llama --container_stop_file container_stop.txt --random_seed 42 --num_iterations 100 --use_prompt_formatting --add_special_tokens
# /dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm
sudo docker cp c71f5b2d5461:/app/tensorrt_llm/examples/llama/outputs/llama/7B/fp16_wq4/1-gpu-24-batch/bmark_numreqsample0_iter100_max1000_v10032gb.out outputs/llama/7B/fp16_wq4/1-gpu-24-batch/bmark_numreqsample0_iter100_max1000_v10032gb.out
