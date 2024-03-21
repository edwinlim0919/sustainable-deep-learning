#!/usr/bin/env bash
# Installation script for some basic bash needs

sudo apt-get update
sudo apt-get install vim
sudo apt-get install tmux

CURR_DIR=$(pwd)
cp ${CURR_DIR}/dotfiles/bashrc ~/.bashrc
cp ${CURR_DIR}/dotfiles/bash_prompt ~/.bash_prompt
cp ${CURR_DIR}/dotfiles/vimrc ~/.vimrc
cp ${CURR_DIR}/dotfiles/tmux.conf ~/.tmux.conf
cp ${CURR_DIR}/dotfiles/sshrc ~/.ssh/rc
cp ${CURR_DIR}/dotfiles/sshconfig ~/.ssh/config
