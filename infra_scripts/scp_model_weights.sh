#!/usr/bin/env bash
# Script to upload model weights to another machine through scp
# Run this script in the same directory as locally stored weights
#
# Arguments
#   args[0] : <user>@<address> for scp/ssh to remote machine
#   args[1] : absolute path to weights folder destination in remote machine
#   args[2] : name of first weights folder
#   ...
#   args[n] : name of last weights folder
#
# Example Usage
#   ./scp_model_weights.sh edwinlim@clnode219.clemson.cloudlab.us /dev/shm/hardware-systems-sustainability/sustainable-deep-learning/model_weights llama-2-7b-chat llama-2-13b-chat llama-2-70b-chat


if [ "$#" -lt 3 ]; then
    echo "Usage: $0 <user>@<address> <destination_path> <folder1> ... <folderN>"
    exit 1
fi


remote_user_host=$1
destination_path=$2
echo "Remote user@host: $remote_user_host"
echo "Destination path: $destination_path"


for folder in "${@:3}"; do
  echo $folder
  scp -r ${folder} ${remote_user_host}:${destination_path}/${folder}
done
