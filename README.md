# Sustainable Deep Learning
Scripting for running deep learning sustainability experiments on various hardware platforms

# Base Setup
First, clone the repository
```bash
cd /dev/shm
git clone git@github.com:edwinlim0919/sustainable-deep-learning.git
cd sustainable-deep-learning
```

Next, set up some bash environment stuff.
You may need to log out of the node and log back in to see env changes take affect.
```bash
yes | source ./env.sh && yes | source ./setup.sh
```

# NVIDIA GPU Setup (CUDA 12.1)
cd /dev/shm/sustainable-deep-learning/nvidia-gpu
```bash
sudo ./remove-nouveau.sh
```
