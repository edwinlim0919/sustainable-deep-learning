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
# Usage
#   ./scp_model_weights.sh edwinlim@clnode219.clemson.cloudlab.us 

#scp -r
