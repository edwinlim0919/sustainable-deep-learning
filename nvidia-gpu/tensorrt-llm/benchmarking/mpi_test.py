from mpi4py import MPI
import os
import subprocess


comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'

print(f"Rank: {rank}, Size: {size}, Visible GPUs: {os.environ.get('CUDA_VISIBLE_DEVICES')}")
