# TensorRT-LLM standalone
sed -i "s/eb316aafb619/b250d4dd5d36/g" cmd_paste_Falcon40B_v10032gb.sh



# ---------- SETTING UP MULTI-NODE MPI ----------
# Find IP address of head node
hostname -I

# Initialize Docker Swarm on the head node
docker swarm init --advertise-addr 130.127.134.25

#cat /etc/hosts # Both nodes in your experiment should be listed here (e.g. node0, node1, ...)
sudo docker swarm join --token SWMTKN-1-2gkral686j38dapepynjo7oz08xl2y21uzj7g6nwracyofmqw8-bfqoc8mhtrasa5tak15asiyxn 130.127.134.25:2377
