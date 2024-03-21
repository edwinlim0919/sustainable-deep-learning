#!/usr/bin/env bash

# Installation flow (c4130 V100 example)
#   sudo ./remove-nouveau.sh
#   sudo ./nvidia-setup.sh https://developer.download.nvidia.com/compute/cuda/12.1.0/local_installers/cuda_12.1.0_530.30.02_linux.run cuda_12.1.0_530.30.02_linux.run

sudo touch /etc/modprobe.d/blacklist-nouveau.conf
sudo echo "blacklist nouveau" >> /etc/modprobe.d/blacklist-nouveau.conf
sudo echo "options nouveau modeset=0" >> /etc/modprobe.d/blacklist-nouveau.conf
sudo update-initramfs -u

# TODO: not sure if explicit command line reboot works, may have to do reboot from cloudlab
sudo reboot
