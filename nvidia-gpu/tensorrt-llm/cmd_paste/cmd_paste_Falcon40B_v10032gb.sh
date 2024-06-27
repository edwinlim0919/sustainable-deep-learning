# TensorRT-LLM standalone
sed -i "s/eb316aafb619/b250d4dd5d36/g" cmd_paste_Falcon40B_v10032gb.sh

# Setting up multi-node MPI
cat /etc/hosts # Both nodes in your experiment should be listed here (e.g. node0, node1, ...)
