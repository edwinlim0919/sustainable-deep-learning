from mpi4py import MPI
import os

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

print(f"Rank: {rank}, Size: {size}, Visible GPUs: {os.environ.get('CUDA_VISIBLE_DEVICES')}")
