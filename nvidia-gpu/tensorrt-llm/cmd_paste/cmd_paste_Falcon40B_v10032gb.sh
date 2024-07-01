# TensorRT-LLM standalone
sed -i "s/b250d4dd5d36/b250d4dd5d36/g" cmd_paste_Falcon40B_v10032gb.sh



# ---------- SETTING UP MULTI-NODE MPI ----------
# /dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm
# (HEAD) Find IP address of head node
hostname -I
# (HEAD) Initialize Docker Swarm on the head node
sudo docker swarm init --advertise-addr 130.127.134.25
# (WORKER) Join the Docker Swarm from the worker node(s)
sudo docker swarm join --token SWMTKN-1-2gkral686j38dapepynjo7oz08xl2y21uzj7g6nwracyofmqw8-bfqoc8mhtrasa5tak15asiyxn 130.127.134.25:2377
# (HEAD) Create an overlay network on the head node
sudo docker network create --driver overlay --attachable trtllm_network
# (HEAD + WORKER) Join the overlay network on both the head node and worker node(s)
sudo docker network connect trtllm_network great_williamson
sudo docker network connect trtllm_network dazzling_hofstadter
# (HEAD + WORKER) Verify that all containers are connected to the network
sudo docker network inspect trtllm_network

# /TensorRT-LLM/examples/falcon
# (HEAD) Set up the MPI hostfile
touch hostfile
echo "10.0.1.2 slots=2" >> hostfile
echo "10.0.1.4 slots=2" >> hostfile
# (HEAD) Ensure SSH daemon is running
apt-get install ssh
service ssh start
# (HEAD) Configure passwordless SSH
ssh-keygen -t rsa -b 2048
# (HEAD) Copy the generated key to the other server
cat ~/.ssh/id_rsa.pub
ssh-rsa AAAAB3NzaC1yc2EAAAADAQABAAABAQClHON3vni5zpJXoxh8jA4BOQI98KvbVstYN90y5cg2618XucRzW9iDnxqkmEROsgDZhwL1ebzCOq8RITRO6We++NOzca3brHUA4vKYAi37gXwm8J1pSUhUF6YYm0KoB9as55wef/b0b6KzFw7/HjDhQz/8roCM+Q1NuYr/7lS2SQnzpNXdnk4T/HY+c5rDQ2m+c4RAOP1AKg6iUuV+yDfHTv/A41Fl63Q8T1Ao2pkmAKCuPXX7yoCMBH74YH0voylpAUa65lpbH9UesG6ncoSvmJh5LmllhicgYFp/0tu5eAI1r3Q25/VHSeClP2YWZRLBXqNouI73DjQIlhrq3gT1 root@b250d4dd5d36
# (WORKER) Create .ssh directory and add public key of HEAD node
mkdir -p ~/.ssh
echo "ssh-rsa AAAAB3NzaC1yc2EAAAADAQABAAABAQClHON3vni5zpJXoxh8jA4BOQI98KvbVstYN90y5cg2618XucRzW9iDnxqkmEROsgDZhwL1ebzCOq8RITRO6We++NOzca3brHUA4vKYAi37gXwm8J1pSUhUF6YYm0KoB9as55wef/b0b6KzFw7/HjDhQz/8roCM+Q1NuYr/7lS2SQnzpNXdnk4T/HY+c5rDQ2m+c4RAOP1AKg6iUuV+yDfHTv/A41Fl63Q8T1Ao2pkmAKCuPXX7yoCMBH74YH0voylpAUa65lpbH9UesG6ncoSvmJh5LmllhicgYFp/0tu5eAI1r3Q25/VHSeClP2YWZRLBXqNouI73DjQIlhrq3gT1 root@b250d4dd5d36" >> ~/.ssh/authorized_keys
chmod 700 ~/.ssh
chmod 600 ~/.ssh/authorized_keys
# Test out the MPI connection
# (HEAD + WORKER) COPY mpi_test.py TO BOTH DOCKER CONTAINERS IN /TensorRT-LLM/examples/falcon/mpi_test.py
sudo docker cp benchmarking/mpi_test.py b250d4dd5d36:/TensorRT-LLM/examples/falcon/mpi_test.py
mpirun -n 4 --hostfile hostfile --allow-run-as-root --oversubscribe python3 mpi_test.py



# ---------- SETTING UP FALCON 40B ----------
# /TensorRT-LLM/examples/falcon
# (HEAD + WORKER) Install and download some prerequisites for Falcon 40B
pip install -r requirements.txt
apt-get update
apt-get -y install git git-lfs
git lfs install
git clone https://huggingface.co/tiiuae/falcon-40b-instruct falcon/40b-instruct

# /TensorRT-LLM/examples/falcon
# (HEAD + WORKER) 4-way tensor parallelism + 1-way pipeline parallelism
python3 convert_checkpoint.py --model_dir ./falcon/40b-instruct --dtype float16 --output_dir ./falcon/40b-instruct/trt_ckpt/fp16/tp4-pp1/ --tp_size 4 --pp_size 1 --load_by_shard
trtllm-build --checkpoint_dir ./falcon/40b-instruct/trt_ckpt/fp16/tp4-pp1/ --gemm_plugin float16 --gpt_attention_plugin float16 --output_dir ./falcon/40b-instruct/trt_engines/fp16/tp4-pp1-batch1/
export NCCL_DEBUG=INFO
export NCCL_SOCKET_IFNAME=eth1
export NCCL_IB_DISABLE=1
mpirun -np 4 --hostfile hostfile --allow-run-as-root --oversubscribe -x NCCL_DEBUG=INFO -x NCCL_SOCKET_IFNAME=eth1 -x NCCL_IB_DISABLE=1 python3 ../summarize.py --test_trt_llm --hf_model_dir ./falcon/40b-instruct/ --engine_dir ./falcon/40b-instruct/trt_engines/fp16/tp4-pp1-batch1/

# /TensorRT-LLM/examples/falcon
# (HEAD + WORKER) 2-way tensor parallelism + 2-way pipeline parallelism
python3 convert_checkpoint.py --model_dir ./falcon/40b-instruct --dtype float16 --output_dir ./falcon/40b-instruct/trt_ckpt/fp16/tp2-pp2/ --tp_size 2 --pp_size 2 --load_by_shard
trtllm-build --checkpoint_dir ./falcon/40b-instruct/trt_ckpt/fp16/tp2-pp2/ --gemm_plugin float16 --gpt_attention_plugin float16 --output_dir ./falcon/40b-instruct/trt_engines/fp16/tp2-pp2-batch1/
export NCCL_DEBUG=INFO
export NCCL_SOCKET_IFNAME=eth1
export NCCL_IB_DISABLE=1
mpirun -np 4 --hostfile hostfile --allow-run-as-root --oversubscribe -x NCCL_DEBUG=INFO -x NCCL_SOCKET_IFNAME=eth1 -x NCCL_IB_DISABLE=1 python3 ../summarize.py --test_trt_llm --hf_model_dir ./falcon/40b-instruct/ --engine_dir ./falcon/40b-instruct/trt_engines/fp16/tp2-pp2-batch1/



# ---------- DEV TESTING ----------
# Testing the actual benchmarking script of multi-gpu multi-node setup
# /dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm
sudo docker cp benchmarking/benchmark_trtllm.py b250d4dd5d36:/TensorRT-LLM/examples/benchmark_trtllm.py
sudo docker cp benchmarking/benchmark_utils.py b250d4dd5d36:/TensorRT-LLM/examples/benchmark_utils.py
sudo docker cp ShareGPT_V3_unfiltered_cleaned_split.json b250d4dd5d36:/TensorRT-LLM/examples/ShareGPT_V3_unfiltered_cleaned_split.json
# /TensorRT-LLM/examples/falcon
mpirun -np 4 --hostfile hostfile --allow-run-as-root --oversubscribe -x NCCL_DEBUG=INFO -x NCCL_SOCKET_IFNAME=eth1 -x NCCL_IB_DISABLE=1 python3 ../benchmark_trtllm.py --tokenizer_dir ./falcon/40b-instruct/ --engine_dir ./falcon/40b-instruct/trt_engines/fp16/tp2-pp2-batch1/ --dataset_path ../ShareGPT_V3_unfiltered_cleaned_split.json --num_requests_sample 0 --max_batch_size 1 --max_input_tokens 1000 --max_output_tokens 1000 --output_dir /TensorRT-LLM/examples/falcon/outputs/40B/fp16/tp2-pp2-batch1/ --output_file bmark_numreqsample0_iter40_max1000_v10032gb.out --container_output_dir /TensorRT-LLM/examples/falcon/ --container_stop_file container_stop.txt --random_seed 42 --num_iterations 10

# Testing multi-node NVSMI script
# /dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm
python3 benchmarking/nvsmi_monitor.py --output_dir ./outputs/falcon/40B/fp16/tp2-pp2-batch1/ --output_file nvsmi_numreqsample0_iter40_max1000_v10032gb.out --container_id b250d4dd5d36 --container_output_dir /TensorRT-LLM/examples/falcon/ --container_stop_file container_stop.txt --gpu_type v10032gb --multi_node --worker_ips 130.127.134.35 --ssh_username edwinlim



# ---------- EXPERIMENT COMMANDS ----------
# max batch size 1
# /TensorRT-LLM/examples/falcon
trtllm-build --checkpoint_dir ./falcon/40b-instruct/trt_ckpt/fp16/tp2-pp2/ --gemm_plugin float16 --gpt_attention_plugin float16 --output_dir ./falcon/40b-instruct/trt_engines/fp16/tp2-pp2-batch1/ --workers 2 --max_batch_size 1
# /dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm
python3 benchmarking/nvsmi_monitor.py --output_dir ./outputs/falcon/40B/fp16/tp2-pp2-batch1/ --output_file nvsmi_numreqsample0_iter50_max1000_v10032gb.out --container_id b250d4dd5d36 --container_output_dir /TensorRT-LLM/examples/falcon/ --container_stop_file container_stop.txt --gpu_type v10032gb --multi_node --worker_ips 130.127.134.35 --ssh_username edwinlim
# /TensorRT-LLM/examples/falcon
mpirun -np 4 --hostfile hostfile --allow-run-as-root --oversubscribe -x NCCL_DEBUG=INFO -x NCCL_SOCKET_IFNAME=eth1 -x NCCL_IB_DISABLE=1 python3 ../benchmark_trtllm.py --tokenizer_dir ./falcon/40b-instruct/ --engine_dir ./falcon/40b-instruct/trt_engines/fp16/tp2-pp2-batch1/ --dataset_path ../ShareGPT_V3_unfiltered_cleaned_split.json --num_requests_sample 0 --max_batch_size 1 --max_input_tokens 1000 --max_output_tokens 1000 --output_dir /TensorRT-LLM/examples/falcon/outputs/40B/fp16/tp2-pp2-batch1/ --output_file bmark_numreqsample0_iter50_max1000_v10032gb.out --container_output_dir /TensorRT-LLM/examples/falcon/ --container_stop_file container_stop.txt --random_seed 42 --num_iterations 50
# /dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm
sudo docker cp b250d4dd5d36:/TensorRT-LLM/examples/falcon/outputs/40B/fp16/tp2-pp2-batch1/bmark_numreqsample0_iter50_max1000_v10032gb.out ./outputs/falcon/40B/fp16/tp2-pp2-batch1/bmark_numreqsample0_iter50_max1000_v10032gb.out
# /TensorRT-LLM/examples/falcon
rm -rf ./falcon/40b-instruct/trt_engines/fp16/tp2-pp2-batch1/

# max batch size 8
# /TensorRT-LLM/examples/falcon
trtllm-build --checkpoint_dir ./falcon/40b-instruct/trt_ckpt/fp16/tp2-pp2/ --gemm_plugin float16 --gpt_attention_plugin float16 --output_dir ./falcon/40b-instruct/trt_engines/fp16/tp2-pp2-batch8/ --workers 2 --max_batch_size 8
# /dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm
python3 benchmarking/nvsmi_monitor.py --output_dir ./outputs/falcon/40B/fp16/tp2-pp2-batch8/ --output_file nvsmi_numreqsample0_iter50_max1000_v10032gb.out --container_id b250d4dd5d36 --container_output_dir /TensorRT-LLM/examples/falcon/ --container_stop_file container_stop.txt --gpu_type v10032gb --multi_node --worker_ips 130.127.134.35 --ssh_username edwinlim
# /TensorRT-LLM/examples/falcon
mpirun -np 4 --hostfile hostfile --allow-run-as-root --oversubscribe -x NCCL_DEBUG=INFO -x NCCL_SOCKET_IFNAME=eth1 -x NCCL_IB_DISABLE=1 python3 ../benchmark_trtllm.py --tokenizer_dir ./falcon/40b-instruct/ --engine_dir ./falcon/40b-instruct/trt_engines/fp16/tp2-pp2-batch8/ --dataset_path ../ShareGPT_V3_unfiltered_cleaned_split.json --num_requests_sample 0 --max_batch_size 8 --max_input_tokens 1000 --max_output_tokens 1000 --output_dir /TensorRT-LLM/examples/falcon/outputs/40B/fp16/tp2-pp2-batch8/ --output_file bmark_numreqsample0_iter50_max1000_v10032gb.out --container_output_dir /TensorRT-LLM/examples/falcon/ --container_stop_file container_stop.txt --random_seed 42 --num_iterations 50
# /dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm
sudo docker cp b250d4dd5d36:/TensorRT-LLM/examples/falcon/outputs/40B/fp16/tp2-pp2-batch8/bmark_numreqsample0_iter50_max1000_v10032gb.out ./outputs/falcon/40B/fp16/tp2-pp2-batch8/bmark_numreqsample0_iter50_max1000_v10032gb.out
# /TensorRT-LLM/examples/falcon
rm -rf ./falcon/40b-instruct/trt_engines/fp16/tp2-pp2-batch8/

# max batch size 16
# /TensorRT-LLM/examples/falcon
trtllm-build --checkpoint_dir ./falcon/40b-instruct/trt_ckpt/fp16/tp2-pp2/ --gemm_plugin float16 --gpt_attention_plugin float16 --output_dir ./falcon/40b-instruct/trt_engines/fp16/tp2-pp2-batch16/ --workers 2 --max_batch_size 16
# /dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm
python3 benchmarking/nvsmi_monitor.py --output_dir ./outputs/falcon/40B/fp16/tp2-pp2-batch16/ --output_file nvsmi_numreqsample0_iter50_max1000_v10032gb.out --container_id b250d4dd5d36 --container_output_dir /TensorRT-LLM/examples/falcon/ --container_stop_file container_stop.txt --gpu_type v10032gb --multi_node --worker_ips 130.127.134.35 --ssh_username edwinlim
# /TensorRT-LLM/examples/falcon
mpirun -np 4 --hostfile hostfile --allow-run-as-root --oversubscribe -x NCCL_DEBUG=INFO -x NCCL_SOCKET_IFNAME=eth1 -x NCCL_IB_DISABLE=1 python3 ../benchmark_trtllm.py --tokenizer_dir ./falcon/40b-instruct/ --engine_dir ./falcon/40b-instruct/trt_engines/fp16/tp2-pp2-batch16/ --dataset_path ../ShareGPT_V3_unfiltered_cleaned_split.json --num_requests_sample 0 --max_batch_size 16 --max_input_tokens 1000 --max_output_tokens 1000 --output_dir /TensorRT-LLM/examples/falcon/outputs/40B/fp16/tp2-pp2-batch16/ --output_file bmark_numreqsample0_iter50_max1000_v10032gb.out --container_output_dir /TensorRT-LLM/examples/falcon/ --container_stop_file container_stop.txt --random_seed 42 --num_iterations 50
# /dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm
sudo docker cp b250d4dd5d36:/TensorRT-LLM/examples/falcon/outputs/40B/fp16/tp2-pp2-batch16/bmark_numreqsample0_iter50_max1000_v10032gb.out ./outputs/falcon/40B/fp16/tp2-pp2-batch16/bmark_numreqsample0_iter50_max1000_v10032gb.out
# /TensorRT-LLM/examples/falcon
rm -rf ./falcon/40b-instruct/trt_engines/fp16/tp2-pp2-batch16/

# max batch size 24
# /TensorRT-LLM/examples/falcon
trtllm-build --checkpoint_dir ./falcon/40b-instruct/trt_ckpt/fp16/tp2-pp2/ --gemm_plugin float16 --gpt_attention_plugin float16 --output_dir ./falcon/40b-instruct/trt_engines/fp16/tp2-pp2-batch24/ --workers 2 --max_batch_size 24
# /dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm
python3 benchmarking/nvsmi_monitor.py --output_dir ./outputs/falcon/40B/fp16/tp2-pp2-batch24/ --output_file nvsmi_numreqsample0_iter50_max1000_v10032gb.out --container_id b250d4dd5d36 --container_output_dir /TensorRT-LLM/examples/falcon/ --container_stop_file container_stop.txt --gpu_type v10032gb --multi_node --worker_ips 130.127.134.35 --ssh_username edwinlim
# /TensorRT-LLM/examples/falcon
mpirun -np 4 --hostfile hostfile --allow-run-as-root --oversubscribe -x NCCL_DEBUG=INFO -x NCCL_SOCKET_IFNAME=eth1 -x NCCL_IB_DISABLE=1 python3 ../benchmark_trtllm.py --tokenizer_dir ./falcon/40b-instruct/ --engine_dir ./falcon/40b-instruct/trt_engines/fp16/tp2-pp2-batch24/ --dataset_path ../ShareGPT_V3_unfiltered_cleaned_split.json --num_requests_sample 0 --max_batch_size 24 --max_input_tokens 1000 --max_output_tokens 1000 --output_dir /TensorRT-LLM/examples/falcon/outputs/40B/fp16/tp2-pp2-batch24/ --output_file bmark_numreqsample0_iter50_max1000_v10032gb.out --container_output_dir /TensorRT-LLM/examples/falcon/ --container_stop_file container_stop.txt --random_seed 42 --num_iterations 50
# /dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm
sudo docker cp b250d4dd5d36:/TensorRT-LLM/examples/falcon/outputs/40B/fp16/tp2-pp2-batch24/bmark_numreqsample0_iter50_max1000_v10032gb.out ./outputs/falcon/40B/fp16/tp2-pp2-batch24/bmark_numreqsample0_iter50_max1000_v10032gb.out
# /TensorRT-LLM/examples/falcon
rm -rf ./falcon/40b-instruct/trt_engines/fp16/tp2-pp2-batch24/

# max batch size 32
# /TensorRT-LLM/examples/falcon
trtllm-build --checkpoint_dir ./falcon/40b-instruct/trt_ckpt/fp16/tp2-pp2/ --gemm_plugin float16 --gpt_attention_plugin float16 --output_dir ./falcon/40b-instruct/trt_engines/fp16/tp2-pp2-batch32/ --workers 2 --max_batch_size 32
# /dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm
python3 benchmarking/nvsmi_monitor.py --output_dir ./outputs/falcon/40B/fp16/tp2-pp2-batch32/ --output_file nvsmi_numreqsample0_iter50_max1000_v10032gb.out --container_id b250d4dd5d36 --container_output_dir /TensorRT-LLM/examples/falcon/ --container_stop_file container_stop.txt --gpu_type v10032gb --multi_node --worker_ips 130.127.134.35 --ssh_username edwinlim
# /TensorRT-LLM/examples/falcon
mpirun -np 4 --hostfile hostfile --allow-run-as-root --oversubscribe -x NCCL_DEBUG=INFO -x NCCL_SOCKET_IFNAME=eth1 -x NCCL_IB_DISABLE=1 python3 ../benchmark_trtllm.py --tokenizer_dir ./falcon/40b-instruct/ --engine_dir ./falcon/40b-instruct/trt_engines/fp16/tp2-pp2-batch32/ --dataset_path ../ShareGPT_V3_unfiltered_cleaned_split.json --num_requests_sample 0 --max_batch_size 32 --max_input_tokens 1000 --max_output_tokens 1000 --output_dir /TensorRT-LLM/examples/falcon/outputs/40B/fp16/tp2-pp2-batch32/ --output_file bmark_numreqsample0_iter50_max1000_v10032gb.out --container_output_dir /TensorRT-LLM/examples/falcon/ --container_stop_file container_stop.txt --random_seed 42 --num_iterations 50
# /dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm
sudo docker cp b250d4dd5d36:/TensorRT-LLM/examples/falcon/outputs/40B/fp16/tp2-pp2-batch32/bmark_numreqsample0_iter50_max1000_v10032gb.out ./outputs/falcon/40B/fp16/tp2-pp2-batch32/bmark_numreqsample0_iter50_max1000_v10032gb.out
# /TensorRT-LLM/examples/falcon
rm -rf ./falcon/40b-instruct/trt_engines/fp16/tp2-pp2-batch32/
