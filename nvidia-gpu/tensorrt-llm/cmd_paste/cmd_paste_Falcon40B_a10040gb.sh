# TensorRT-LLM standalone
sed -i "s/eb316aafb619/eb316aafb619/g" cmd_paste_Falcon40B_a10040gb.sh

# /TensorRT-LLM/examples/falcon
pip install -r requirements.txt
apt-get update
apt-get -y install git git-lfs
git lfs install
git clone https://huggingface.co/tiiuae/falcon-40b-instruct falcon/40b-instruct

# /dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm
sudo docker cp benchmarking/benchmark_trtllm.py eb316aafb619:/TensorRT-LLM/examples/benchmark_trtllm.py
sudo docker cp benchmarking/benchmark_utils.py eb316aafb619:/TensorRT-LLM/examples/benchmark_utils.py
sudo docker cp ShareGPT_V3_unfiltered_cleaned_split.json eb316aafb619:/TensorRT-LLM/examples/ShareGPT_V3_unfiltered_cleaned_split.json
sudo docker cp ShareGPT_V3_unfiltered_cleaned_split_top100.json eb316aafb619:/TensorRT-LLM/examples/ShareGPT_V3_unfiltered_cleaned_split_top100.json


# DEV TESTING
# /TensorRT-LLM/examples/falcon
sudo docker cp eb316aafb619:/TensorRT-LLM/examples/summarize.py benchmarking/summarize.py
mpirun -n 4 --allow-run-as-root --oversubscribe python3 ../summarize.py --test_trt_llm --hf_model_dir ./falcon/40b-instruct/ --engine_dir ./falcon/40b-instruct/trt_engines/bf16/tp4-pp1-batch1/

# /dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm
python3 benchmarking/nvsmi_monitor.py --output_dir ./outputs/falcon/40B/bf16/tp4-pp1-batch1/ --output_file nvsmi_numreqsample0_iter100_max1000_a10040gb.out --container_id eb316aafb619 --container_output_dir /TensorRT-LLM/examples/falcon/ --container_stop_file container_stop.txt --gpu_type a10040gb
# /TensorRT-LLM/examples/falcon
mpirun -n 4 --allow-run-as-root --oversubscribe python3 ../benchmark_trtllm.py --tokenizer_dir ./falcon/40b-instruct/ --engine_dir ./falcon/40b-instruct/trt_engines/bf16/tp4-pp1-batch1/ --dataset_path ../ShareGPT_V3_unfiltered_cleaned_split.json --num_requests_sample 0 --max_batch_size 1 --max_input_tokens 1000 --max_output_tokens 1000 --output_dir /TensorRT-LLM/examples/falcon/outputs/40B/bf16/tp4-pp1-batch1/ --output_file bmark_numreqsample0_iter100_max1000_a10040gb.out --container_output_dir /TensorRT-LLM/examples/falcon/ --container_stop_file container_stop.txt --random_seed 42 --num_iterations 10


# /TensorRT-LLM/examples/falcon
# 4-way tensor parallelism + 1-way pipeline parallelism
python3 convert_checkpoint.py --model_dir ./falcon/40b-instruct --dtype bfloat16 --output_dir ./falcon/40b-instruct/trt_ckpt/bf16/tp4-pp1/ --tp_size 4 --pp_size 1 --load_by_shard
rm -rf ./falcon/40b-instruct/trt_ckpt/bf16/tp4-pp1/

# max_batch_size 1
# /TensorRT-LLM/examples/falcon
trtllm-build --checkpoint_dir ./falcon/40b-instruct/trt_ckpt/bf16/tp4-pp1/ --gemm_plugin bfloat16 --gpt_attention_plugin bfloat16 --output_dir ./falcon/40b-instruct/trt_engines/bf16/tp4-pp1-batch1/ --workers 4 --max_batch_size 1
# /dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm
python3 benchmarking/nvsmi_monitor.py --output_dir ./outputs/falcon/40B/bf16/tp4-pp1-batch1/ --output_file nvsmi_numreqsample0_iter100_max1000_a10040gb.out --container_id eb316aafb619 --container_output_dir /TensorRT-LLM/examples/falcon/ --container_stop_file container_stop.txt --gpu_type a10040gb
# /TensorRT-LLM/examples/falcon
mpirun -n 4 --allow-run-as-root --oversubscribe python3 ../benchmark_trtllm.py --tokenizer_dir ./falcon/40b-instruct/ --engine_dir ./falcon/40b-instruct/trt_engines/bf16/tp4-pp1-batch1/ --dataset_path ../ShareGPT_V3_unfiltered_cleaned_split.json --num_requests_sample 0 --max_batch_size 1 --max_input_tokens 1000 --max_output_tokens 1000 --output_dir /TensorRT-LLM/examples/falcon/outputs/40B/bf16/tp4-pp1-batch1/ --output_file bmark_numreqsample0_iter100_max1000_a10040gb.out --container_output_dir /TensorRT-LLM/examples/falcon/ --container_stop_file container_stop.txt --random_seed 42 --num_iterations 100
# /dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm
sudo docker cp eb316aafb619:/TensorRT-LLM/examples/falcon/outputs/40B/bf16/tp4-pp1-batch1/bmark_numreqsample0_iter100_max1000_a10040gb.out ./outputs/falcon/40B/bf16/tp4-pp1-batch1/bmark_numreqsample0_iter100_max1000_a10040gb.out
# /TensorRT-LLM/examples/falcon
rm -rf ./falcon/40b-instruct/trt_engines/bf16/tp4-pp1-batch1/

# max_batch_size 8
# /TensorRT-LLM/examples/falcon
trtllm-build --checkpoint_dir ./falcon/40b-instruct/trt_ckpt/bf16/tp4-pp1/ --gemm_plugin bfloat16 --gpt_attention_plugin bfloat16 --output_dir ./falcon/40b-instruct/trt_engines/bf16/tp4-pp1-batch8/ --workers 4 --max_batch_size 8
# /dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm
python3 benchmarking/nvsmi_monitor.py --output_dir ./outputs/falcon/40B/bf16/tp4-pp1-batch8/ --output_file nvsmi_numreqsample0_iter100_max1000_a10040gb.out --container_id eb316aafb619 --container_output_dir /TensorRT-LLM/examples/falcon/ --container_stop_file container_stop.txt --gpu_type a10040gb
# /TensorRT-LLM/examples/falcon
mpirun -n 4 --allow-run-as-root --oversubscribe python3 ../benchmark_trtllm.py --tokenizer_dir ./falcon/40b-instruct/ --engine_dir ./falcon/40b-instruct/trt_engines/bf16/tp4-pp1-batch8/ --dataset_path ../ShareGPT_V3_unfiltered_cleaned_split.json --num_requests_sample 0 --max_batch_size 8 --max_input_tokens 1000 --max_output_tokens 1000 --output_dir /TensorRT-LLM/examples/falcon/outputs/40B/bf16/tp4-pp1-batch8/ --output_file bmark_numreqsample0_iter100_max1000_a10040gb.out --container_output_dir /TensorRT-LLM/examples/falcon/ --container_stop_file container_stop.txt --random_seed 42 --num_iterations 100
# /dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm
sudo docker cp eb316aafb619:/TensorRT-LLM/examples/falcon/outputs/40B/bf16/tp4-pp1-batch8/bmark_numreqsample0_iter100_max1000_a10040gb.out ./outputs/falcon/40B/bf16/tp4-pp1-batch8/bmark_numreqsample0_iter100_max1000_a10040gb.out
# /TensorRT-LLM/examples/falcon
rm -rf ./falcon/40b-instruct/trt_engines/bf16/tp4-pp1-batch8/

# max_batch_size 16
# /TensorRT-LLM/examples/falcon
trtllm-build --checkpoint_dir ./falcon/40b-instruct/trt_ckpt/bf16/tp4-pp1/ --gemm_plugin bfloat16 --gpt_attention_plugin bfloat16 --output_dir ./falcon/40b-instruct/trt_engines/bf16/tp4-pp1-batch16/ --workers 4 --max_batch_size 16
# /dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm
python3 benchmarking/nvsmi_monitor.py --output_dir ./outputs/falcon/40B/bf16/tp4-pp1-batch16/ --output_file nvsmi_numreqsample0_iter100_max1000_a10040gb.out --container_id eb316aafb619 --container_output_dir /TensorRT-LLM/examples/falcon/ --container_stop_file container_stop.txt --gpu_type a10040gb
# /TensorRT-LLM/examples/falcon
mpirun -n 4 --allow-run-as-root --oversubscribe python3 ../benchmark_trtllm.py --tokenizer_dir ./falcon/40b-instruct/ --engine_dir ./falcon/40b-instruct/trt_engines/bf16/tp4-pp1-batch16/ --dataset_path ../ShareGPT_V3_unfiltered_cleaned_split.json --num_requests_sample 0 --max_batch_size 16 --max_input_tokens 1000 --max_output_tokens 1000 --output_dir /TensorRT-LLM/examples/falcon/outputs/40B/bf16/tp4-pp1-batch16/ --output_file bmark_numreqsample0_iter100_max1000_a10040gb.out --container_output_dir /TensorRT-LLM/examples/falcon/ --container_stop_file container_stop.txt --random_seed 42 --num_iterations 100
# /dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm
sudo docker cp eb316aafb619:/TensorRT-LLM/examples/falcon/outputs/40B/bf16/tp4-pp1-batch16/bmark_numreqsample0_iter100_max1000_a10040gb.out ./outputs/falcon/40B/bf16/tp4-pp1-batch16/bmark_numreqsample0_iter100_max1000_a10040gb.out
# /TensorRT-LLM/examples/falcon
rm -rf ./falcon/40b-instruct/trt_engines/bf16/tp4-pp1-batch16/

# max_batch_size 24
# /TensorRT-LLM/examples/falcon
trtllm-build --checkpoint_dir ./falcon/40b-instruct/trt_ckpt/bf16/tp4-pp1/ --gemm_plugin bfloat16 --gpt_attention_plugin bfloat16 --output_dir ./falcon/40b-instruct/trt_engines/bf16/tp4-pp1-batch24/ --workers 4 --max_batch_size 24
# /dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm
python3 benchmarking/nvsmi_monitor.py --output_dir ./outputs/falcon/40B/bf16/tp4-pp1-batch24/ --output_file nvsmi_numreqsample0_iter100_max1000_a10040gb.out --container_id eb316aafb619 --container_output_dir /TensorRT-LLM/examples/falcon/ --container_stop_file container_stop.txt --gpu_type a10040gb
# /TensorRT-LLM/examples/falcon
mpirun -n 4 --allow-run-as-root --oversubscribe python3 ../benchmark_trtllm.py --tokenizer_dir ./falcon/40b-instruct/ --engine_dir ./falcon/40b-instruct/trt_engines/bf16/tp4-pp1-batch24/ --dataset_path ../ShareGPT_V3_unfiltered_cleaned_split.json --num_requests_sample 0 --max_batch_size 24 --max_input_tokens 1000 --max_output_tokens 1000 --output_dir /TensorRT-LLM/examples/falcon/outputs/40B/bf16/tp4-pp1-batch24/ --output_file bmark_numreqsample0_iter100_max1000_a10040gb.out --container_output_dir /TensorRT-LLM/examples/falcon/ --container_stop_file container_stop.txt --random_seed 42 --num_iterations 100
# /dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm
sudo docker cp eb316aafb619:/TensorRT-LLM/examples/falcon/outputs/40B/bf16/tp4-pp1-batch24/bmark_numreqsample0_iter100_max1000_a10040gb.out ./outputs/falcon/40B/bf16/tp4-pp1-batch24/bmark_numreqsample0_iter100_max1000_a10040gb.out
# /TensorRT-LLM/examples/falcon
rm -rf ./falcon/40b-instruct/trt_engines/bf16/tp4-pp1-batch24/

# max_batch_size 32
# /TensorRT-LLM/examples/falcon
trtllm-build --checkpoint_dir ./falcon/40b-instruct/trt_ckpt/bf16/tp4-pp1/ --gemm_plugin bfloat16 --gpt_attention_plugin bfloat16 --output_dir ./falcon/40b-instruct/trt_engines/bf16/tp4-pp1-batch32/ --workers 4 --max_batch_size 32
# /dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm
python3 benchmarking/nvsmi_monitor.py --output_dir ./outputs/falcon/40B/bf16/tp4-pp1-batch32/ --output_file nvsmi_numreqsample0_iter100_max1000_a10040gb.out --container_id eb316aafb619 --container_output_dir /TensorRT-LLM/examples/falcon/ --container_stop_file container_stop.txt --gpu_type a10040gb
# /TensorRT-LLM/examples/falcon
mpirun -n 4 --allow-run-as-root --oversubscribe python3 ../benchmark_trtllm.py --tokenizer_dir ./falcon/40b-instruct/ --engine_dir ./falcon/40b-instruct/trt_engines/bf16/tp4-pp1-batch32/ --dataset_path ../ShareGPT_V3_unfiltered_cleaned_split.json --num_requests_sample 0 --max_batch_size 32 --max_input_tokens 1000 --max_output_tokens 1000 --output_dir /TensorRT-LLM/examples/falcon/outputs/40B/bf16/tp4-pp1-batch32/ --output_file bmark_numreqsample0_iter100_max1000_a10040gb.out --container_output_dir /TensorRT-LLM/examples/falcon/ --container_stop_file container_stop.txt --random_seed 42 --num_iterations 100
# /dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm
sudo docker cp eb316aafb619:/TensorRT-LLM/examples/falcon/outputs/40B/bf16/tp4-pp1-batch32/bmark_numreqsample0_iter100_max1000_a10040gb.out ./outputs/falcon/40B/bf16/tp4-pp1-batch32/bmark_numreqsample0_iter100_max1000_a10040gb.out
# /TensorRT-LLM/examples/falcon
rm -rf ./falcon/40b-instruct/trt_engines/bf16/tp4-pp1-batch32/

# max_batch_size 40
# /TensorRT-LLM/examples/falcon
trtllm-build --checkpoint_dir ./falcon/40b-instruct/trt_ckpt/bf16/tp4-pp1/ --gemm_plugin bfloat16 --gpt_attention_plugin bfloat16 --output_dir ./falcon/40b-instruct/trt_engines/bf16/tp4-pp1-batch40/ --workers 4 --max_batch_size 40
# /dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm
python3 benchmarking/nvsmi_monitor.py --output_dir ./outputs/falcon/40B/bf16/tp4-pp1-batch40/ --output_file nvsmi_numreqsample0_iter100_max1000_a10040gb.out --container_id eb316aafb619 --container_output_dir /TensorRT-LLM/examples/falcon/ --container_stop_file container_stop.txt --gpu_type a10040gb
# /TensorRT-LLM/examples/falcon
mpirun -n 4 --allow-run-as-root --oversubscribe python3 ../benchmark_trtllm.py --tokenizer_dir ./falcon/40b-instruct/ --engine_dir ./falcon/40b-instruct/trt_engines/bf16/tp4-pp1-batch40/ --dataset_path ../ShareGPT_V3_unfiltered_cleaned_split.json --num_requests_sample 0 --max_batch_size 40 --max_input_tokens 1000 --max_output_tokens 1000 --output_dir /TensorRT-LLM/examples/falcon/outputs/40B/bf16/tp4-pp1-batch40/ --output_file bmark_numreqsample0_iter100_max1000_a10040gb.out --container_output_dir /TensorRT-LLM/examples/falcon/ --container_stop_file container_stop.txt --random_seed 42 --num_iterations 100
# /dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm
sudo docker cp eb316aafb619:/TensorRT-LLM/examples/falcon/outputs/40B/bf16/tp4-pp1-batch40/bmark_numreqsample0_iter100_max1000_a10040gb.out ./outputs/falcon/40B/bf16/tp4-pp1-batch40/bmark_numreqsample0_iter100_max1000_a10040gb.out
# /TensorRT-LLM/examples/falcon
rm -rf ./falcon/40b-instruct/trt_engines/bf16/tp4-pp1-batch40/

# max_batch_size 48
# /TensorRT-LLM/examples/falcon
trtllm-build --checkpoint_dir ./falcon/40b-instruct/trt_ckpt/bf16/tp4-pp1/ --gemm_plugin bfloat16 --gpt_attention_plugin bfloat16 --output_dir ./falcon/40b-instruct/trt_engines/bf16/tp4-pp1-batch48/ --workers 4 --max_batch_size 48
# /dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm
python3 benchmarking/nvsmi_monitor.py --output_dir ./outputs/falcon/40B/bf16/tp4-pp1-batch48/ --output_file nvsmi_numreqsample0_iter100_max1000_a10040gb.out --container_id eb316aafb619 --container_output_dir /TensorRT-LLM/examples/falcon/ --container_stop_file container_stop.txt --gpu_type a10040gb
# /TensorRT-LLM/examples/falcon
mpirun -n 4 --allow-run-as-root --oversubscribe python3 ../benchmark_trtllm.py --tokenizer_dir ./falcon/40b-instruct/ --engine_dir ./falcon/40b-instruct/trt_engines/bf16/tp4-pp1-batch48/ --dataset_path ../ShareGPT_V3_unfiltered_cleaned_split.json --num_requests_sample 0 --max_batch_size 48 --max_input_tokens 1000 --max_output_tokens 1000 --output_dir /TensorRT-LLM/examples/falcon/outputs/40B/bf16/tp4-pp1-batch48/ --output_file bmark_numreqsample0_iter100_max1000_a10040gb.out --container_output_dir /TensorRT-LLM/examples/falcon/ --container_stop_file container_stop.txt --random_seed 42 --num_iterations 100
# /dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm
sudo docker cp eb316aafb619:/TensorRT-LLM/examples/falcon/outputs/40B/bf16/tp4-pp1-batch48/bmark_numreqsample0_iter100_max1000_a10040gb.out ./outputs/falcon/40B/bf16/tp4-pp1-batch48/bmark_numreqsample0_iter100_max1000_a10040gb.out
# /TensorRT-LLM/examples/falcon
rm -rf ./falcon/40b-instruct/trt_engines/bf16/tp4-pp1-batch48/

# max_batch_size 56
# /TensorRT-LLM/examples/falcon
trtllm-build --checkpoint_dir ./falcon/40b-instruct/trt_ckpt/bf16/tp4-pp1/ --gemm_plugin bfloat16 --gpt_attention_plugin bfloat16 --output_dir ./falcon/40b-instruct/trt_engines/bf16/tp4-pp1-batch56/ --workers 4 --max_batch_size 56
# /dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm
python3 benchmarking/nvsmi_monitor.py --output_dir ./outputs/falcon/40B/bf16/tp4-pp1-batch56/ --output_file nvsmi_numreqsample0_iter100_max1000_a10040gb.out --container_id eb316aafb619 --container_output_dir /TensorRT-LLM/examples/falcon/ --container_stop_file container_stop.txt --gpu_type a10040gb
# /TensorRT-LLM/examples/falcon
mpirun -n 4 --allow-run-as-root --oversubscribe python3 ../benchmark_trtllm.py --tokenizer_dir ./falcon/40b-instruct/ --engine_dir ./falcon/40b-instruct/trt_engines/bf16/tp4-pp1-batch56/ --dataset_path ../ShareGPT_V3_unfiltered_cleaned_split.json --num_requests_sample 0 --max_batch_size 56 --max_input_tokens 1000 --max_output_tokens 1000 --output_dir /TensorRT-LLM/examples/falcon/outputs/40B/bf16/tp4-pp1-batch56/ --output_file bmark_numreqsample0_iter100_max1000_a10040gb.out --container_output_dir /TensorRT-LLM/examples/falcon/ --container_stop_file container_stop.txt --random_seed 42 --num_iterations 100
# /dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm
sudo docker cp eb316aafb619:/TensorRT-LLM/examples/falcon/outputs/40B/bf16/tp4-pp1-batch56/bmark_numreqsample0_iter100_max1000_a10040gb.out ./outputs/falcon/40B/bf16/tp4-pp1-batch56/bmark_numreqsample0_iter100_max1000_a10040gb.out
# /TensorRT-LLM/examples/falcon
rm -rf ./falcon/40b-instruct/trt_engines/bf16/tp4-pp1-batch56/

# max_batch_size 64
# /TensorRT-LLM/examples/falcon
trtllm-build --checkpoint_dir ./falcon/40b-instruct/trt_ckpt/bf16/tp4-pp1/ --gemm_plugin bfloat16 --gpt_attention_plugin bfloat16 --output_dir ./falcon/40b-instruct/trt_engines/bf16/tp4-pp1-batch64/ --workers 4 --max_batch_size 64
# /dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm
python3 benchmarking/nvsmi_monitor.py --output_dir ./outputs/falcon/40B/bf16/tp4-pp1-batch64/ --output_file nvsmi_numreqsample0_iter100_max1000_a10040gb.out --container_id eb316aafb619 --container_output_dir /TensorRT-LLM/examples/falcon/ --container_stop_file container_stop.txt --gpu_type a10040gb
# /TensorRT-LLM/examples/falcon
mpirun -n 4 --allow-run-as-root --oversubscribe python3 ../benchmark_trtllm.py --tokenizer_dir ./falcon/40b-instruct/ --engine_dir ./falcon/40b-instruct/trt_engines/bf16/tp4-pp1-batch64/ --dataset_path ../ShareGPT_V3_unfiltered_cleaned_split.json --num_requests_sample 0 --max_batch_size 64 --max_input_tokens 1000 --max_output_tokens 1000 --output_dir /TensorRT-LLM/examples/falcon/outputs/40B/bf16/tp4-pp1-batch64/ --output_file bmark_numreqsample0_iter100_max1000_a10040gb.out --container_output_dir /TensorRT-LLM/examples/falcon/ --container_stop_file container_stop.txt --random_seed 42 --num_iterations 100
# /dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm
sudo docker cp eb316aafb619:/TensorRT-LLM/examples/falcon/outputs/40B/bf16/tp4-pp1-batch64/bmark_numreqsample0_iter100_max1000_a10040gb.out ./outputs/falcon/40B/bf16/tp4-pp1-batch64/bmark_numreqsample0_iter100_max1000_a10040gb.out
# /TensorRT-LLM/examples/falcon
rm -rf ./falcon/40b-instruct/trt_engines/bf16/tp4-pp1-batch64/



# max_batch_size 80
# /TensorRT-LLM/examples/falcon
trtllm-build --checkpoint_dir ./falcon/40b-instruct/trt_ckpt/bf16/tp4-pp1/ --gemm_plugin bfloat16 --gpt_attention_plugin bfloat16 --output_dir ./falcon/40b-instruct/trt_engines/bf16/tp4-pp1-batch80/ --workers 4 --max_batch_size 80
# /dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm
python3 benchmarking/nvsmi_monitor.py --output_dir ./outputs/falcon/40B/bf16/tp4-pp1-batch80/ --output_file nvsmi_numreqsample0_iter100_max1000_a10040gb.out --container_id eb316aafb619 --container_output_dir /TensorRT-LLM/examples/falcon/ --container_stop_file container_stop.txt --gpu_type a10040gb
# /TensorRT-LLM/examples/falcon
mpirun -n 4 --allow-run-as-root --oversubscribe python3 ../benchmark_trtllm.py --tokenizer_dir ./falcon/40b-instruct/ --engine_dir ./falcon/40b-instruct/trt_engines/bf16/tp4-pp1-batch80/ --dataset_path ../ShareGPT_V3_unfiltered_cleaned_split.json --num_requests_sample 0 --max_batch_size 80 --max_input_tokens 1000 --max_output_tokens 1000 --output_dir /TensorRT-LLM/examples/falcon/outputs/40B/bf16/tp4-pp1-batch80/ --output_file bmark_numreqsample0_iter100_max1000_a10040gb.out --container_output_dir /TensorRT-LLM/examples/falcon/ --container_stop_file container_stop.txt --random_seed 42 --num_iterations 100
# /dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm
sudo docker cp eb316aafb619:/TensorRT-LLM/examples/falcon/outputs/40B/bf16/tp4-pp1-batch80/bmark_numreqsample0_iter100_max1000_a10040gb.out ./outputs/falcon/40B/bf16/tp4-pp1-batch80/bmark_numreqsample0_iter100_max1000_a10040gb.out
# /TensorRT-LLM/examples/falcon
rm -rf ./falcon/40b-instruct/trt_engines/bf16/tp4-pp1-batch80/
