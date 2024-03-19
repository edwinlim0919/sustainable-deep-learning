#!/usr/bin/env bash

# Installation flow
# 

sudo touch /etc/modprobe.d/blacklist-nouveau.conf
sudo echo "blacklist nouveau" >> /etc/modprobe.d/blacklist-nouveau.conf
sudo echo "options nouveau modeset=0" >> /etc/modprobe.d/blacklist-nouveau.conf
sudo update-initramfs -u

# TODO: not sure if explicit command line reboot works, may have to do reboot from cloudlab
sudo reboot
