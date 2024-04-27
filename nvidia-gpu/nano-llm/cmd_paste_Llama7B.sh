sed -i "s/a581d665ef30/a581d665ef30/g" cmd_paste_Llama7B.sh

# container "/"
mkdir nano-llm

# host "/home/edgeml/experiments/sustainable-deep-learning/nvidia-gpu/nano-llm"
sudo docker cp testing_nano-llm.py a581d665ef30:/nano-llm/testing_nano-llm.py
