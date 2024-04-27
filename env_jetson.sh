#!/usr/bin/env bash
# Installation script for some basic bash needs

sudo apt-get update
sudo apt-get install vim
sudo apt-get install tmux

CURR_DIR=$(pwd)
cp ${CURR_DIR}/dotfiles_jetson/bashrc ~/.bashrc
cp ${CURR_DIR}/dotfiles_jetson/bash_prompt ~/.bash_prompt
cp ${CURR_DIR}/dotfiles_jetson/vimrc ~/.vimrc
cp ${CURR_DIR}/dotfiles_jetson/tmux.conf ~/.tmux.conf
cp ${CURR_DIR}/dotfiles_jetson/sshrc ~/.ssh/rc
cp ${CURR_DIR}/dotfiles_jetson/sshconfig ~/.ssh/config
