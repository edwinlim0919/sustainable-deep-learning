# TensorRT-LLM standalone
sed -i "s/eb316aafb619/b250d4dd5d36/g" cmd_paste_Falcon40B_v10032gb.sh


# ---------- SETTING UP MULTI-NODE MPI ----------
# Find IP address of head node
hostname -I

# Initialize Docker Swarm on the head node
sudo docker swarm init --advertise-addr 130.127.134.25

# Join the Docker Swarm from the worker node(s)
sudo docker swarm join --token SWMTKN-1-2gkral686j38dapepynjo7oz08xl2y21uzj7g6nwracyofmqw8-bfqoc8mhtrasa5tak15asiyxn 130.127.134.25:2377

# Create an overlay network on the head node
sudo docker network create --driver overlay --attachable trtllm_network

# Join the overlay network on both the head node and worker node(s)
sudo docker network connect trtllm_network great_williamson
sudo docker network connect trtllm_network dazzling_hofstadter

# Verify that all containers are connected to the network
sudo docker network inspect trtllm_network


# ---------- SETTING UP FALCON 40B ----------
# /TensorRT-LLM/examples/falcon
pip install -r requirements.txt
apt-get update
apt-get -y install git git-lfs
git lfs install
git clone https://huggingface.co/tiiuae/falcon-40b-instruct falcon/40b-instruct

# /TensorRT-LLM/examples/falcon
# 4-way tensor parallelism + 1-way pipeline parallelism
python3 convert_checkpoint.py --model_dir ./falcon/40b-instruct --dtype float16 --output_dir ./falcon/40b-instruct/trt_ckpt/fp16/tp4-pp1/ --tp_size 4 --pp_size 1 --load_by_shard
trtllm-build --checkpoint_dir ./falcon/40b-instruct/trt_ckpt/fp16/tp4-pp1/ --gemm_plugin float16 --gpt_attention_plugin float16 --output_dir ./falcon/40b-instruct/trt_engines/fp16/tp4-pp1-batch1/
mpirun -n 4 --allow-run-as-root --oversubscribe python3 ../summarize.py --test_trt_llm --hf_model_dir ./falcon/40b-instruct/ --engine_dir ./falcon/40b-instruct/trt_engines/fp16/tp4-pp1-batch1/
