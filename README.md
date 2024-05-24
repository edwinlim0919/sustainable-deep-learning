# Sustainable Deep Learning
This repository contains scripts for running deep learning sustainability experiments on various hardware platforms.

# Base Setup
First, clone the repository.
```bash
cd /dev/shm
git clone git@github.com:edwinlim0919/sustainable-deep-learning.git
cd sustainable-deep-learning
```

Finally, set up some bash environment stuff.
Feel free to skip this step if you have your own bash preferences.
You may need to log out of the node and log back in to see env changes take effect.
```bash
yes | source ./env.sh
```

Next, set up some basic dependencies for the project.
Depending on your host machine architecture, install either x86_64 or aarch64.
```bash
# x86_64
yes | source ./setup_x86_64.sh

# aarch64
yes | source ./setup_aarch64.sh
```

# NVIDIA GPU Setup
To use NVIDIA GPUs, remove nouveau from Linux and install NVIDIA CUDA drivers.
## Removing nouveau
```bash
# This will reboot your node and you will need to re-clone this repo in /dev/shm
cd /dev/shm/sustainable-deep-learning/nvidia-gpu
sudo ./remove-nouveau.sh
```

## Installing CUDA
```bash
# This script will reboot your node as well, and you will need to reclone this repo in /dev/shm (examples provided below).
# Install one version of CUDA only.
# Figure out which version of CUDA 12.1 to use for your machine from here: https://developer.nvidia.com/cuda-12-1-0-download-archive
# CUDA 12.1
cd /dev/shm/sustainable-deep-learning/nvidia-gpu
sudo ./nvidia-setup.sh https://developer.download.nvidia.com/compute/cuda/12.1.0/local_installers/cuda_12.1.0_530.30.02_linux.run cuda_12.1.0_530.30.02_linux.run

# Figure out which version of CUDA 12.2 to use for your machine from here: https://developer.nvidia.com/cuda-12-2-0-download-archive
# CUDA 12.2
cd /dev/shm/sustainable-deep-learning/nvidia-gpu
sudo ./nvidia-setup.sh https://developer.download.nvidia.com/compute/cuda/12.2.0/local_installers/cuda_12.2.0_535.54.03_linux.run cuda_12.2.0_535.54.03_linux.run
```

# NVIDIA TensorRT-LLM setup
First, follow the instructions from the link below to increase the size of your Linux root filesystem partition (~200 GB recommended).
https://www.privex.io/articles/how-to-resize-partition/

Then, run the following script to setup TensorRT-LLM.
Do not install CUDA manually when setting up TensorRT-LLM (container is pre-packaged with CUDA).
```bash
cd /dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm
source ./setup_tensorrt-llm.sh
```

# NVIDIA TensorRT setup
Manual CUDA installation (12.2) is required for the TensorRT setup script.


# Running Llama2 7B Experiments
Start the nvidia-persistenced daemon, and then start the NVIDIA docker container.
```bash
sudo nvidia-persistenced --user root
sudo make -C docker release_run
```

NOTE: If you get an error related to some sort of version mismatch, run the following commands and try starting the NVIDIA docker container again.
```bash
sudo apt-get --purge remove "*nvidia*"
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
sudo make -C docker release_run
```

Once the NVIDIA docker container is running, get the PID of the container with the following command. You will need this PID to copy things back and forth from your NVIDIA docker container.
```bash
sudo docker ps # Replace <DOCKER CONTAINER PID> with the PID from this command
```

Download the Llama2 7B weights from HuggingFace and copy them to your docker container.
```bash
huggingface-cli login
python3 download_hf_weights.py --model-name "meta-llama/Llama-2-7b-chat-hf"
sudo docker cp meta-llama/ <DOCKER CONTAINER PID>:/app/tensorrt_llm/examples/llama
```

Within the docker container, install a few dependencies. Run the command from the specified directory within the docker container.
```bash
# /app/tensorrt_llm/examples/llama
pip install nltk
pip install rouge_score
pip install aiofiles
```

Within the docker container, convert the HuggingFace checkpoint to be compatible with TensorRT-LLM, and then build the TRT engine.
```bash
# /app/tensorrt_llm/examples/llama
python convert_checkpoint.py --model_dir meta-llama/Llama-2-7b-chat-hf_model --dtype float16 --output_dir ./llama/7B/trt_ckpt/fp16/1-gpu/
trtllm-build --checkpoint_dir ./llama/7B/trt_ckpt/fp16/1-gpu/ --gemm_plugin float16 --output_dir ./llama/7B/trt_engines/fp16/1-gpu-1-batch/ --max_batch_size 1
```

Within your normal bash environment, copy benchmarking scripts and a mini prompt dataset (full dataset is ShareGPT) to the docker container.
```bash
# /dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm
sudo docker cp benchmarking/benchmark_trtllm.py <DOCKER CONTAINER PID>:/app/tensorrt_llm/examples/benchmark_trtllm.py
sudo docker cp benchmarking/benchmark_utils.py <DOCKER CONTAINER PID>:/app/tensorrt_llm/examples/benchmark_utils.py
sudo docker cp ShareGPT_V3_unfiltered_cleaned_split_top100.json <DOCKER CONTAINER PID>:/app/tensorrt_llm/examples/ShareGPT_V3_unfiltered_cleaned_split_top100.json
```

Finally, run a small benchmark from within the docker container.
```bash
# /app/tensorrt_llm/examples/llama
python ../benchmark_trtllm.py --tokenizer_dir ./meta-llama/Llama-2-7b-chat-hf_tokenizer/ --engine_dir ./llama/7B/trt_engines/fp16/1-gpu-1-batch/ --dataset_path ../ShareGPT_V3_unfiltered_cleaned_split_top100.json --num_requests_sample 4 --max_batch_size 1 --max_input_tokens 1000 --max_output_tokens 1000 --output_dir ./outputs/llama/7B/fp16/1-gpu-1-batch --output_file dev_testing.out --random_seed 42 --num_iterations 10
```
