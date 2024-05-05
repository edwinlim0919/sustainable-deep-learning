python gpu_batch_exp_plotting.py --bmark_output_paths    "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/13B/fp16/1-gpu-1-batch/bmark_numreqsample0_iter100_max1000.out" \
								 					     "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/13B/fp16/1-gpu-1-batch/bmark_numreqsample0_iter100_max500.out" \
													     "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/13B/fp16/1-gpu-2-batch/bmark_numreqsample0_iter100_max1000.out" \
								 					     "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/13B/fp16/1-gpu-2-batch/bmark_numreqsample0_iter100_max500.out" \
													     "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/13B/fp16/1-gpu-4-batch/bmark_numreqsample0_iter100_max1000.out" \
								 					     "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/13B/fp16/1-gpu-4-batch/bmark_numreqsample0_iter100_max500.out" \
								 					     "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/13B/fp16/1-gpu-6-batch/bmark_numreqsample0_iter100_max500.out" \
								 --nvsmi_output_paths    "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/13B/fp16/1-gpu-1-batch/nvsmi_numreqsample0_iter100_max1000.out" \
								 					     "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/13B/fp16/1-gpu-1-batch/nvsmi_numreqsample0_iter100_max500.out" \
													     "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/13B/fp16/1-gpu-2-batch/nvsmi_numreqsample0_iter100_max1000.out" \
								 					     "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/13B/fp16/1-gpu-2-batch/nvsmi_numreqsample0_iter100_max500.out" \
													     "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/13B/fp16/1-gpu-4-batch/nvsmi_numreqsample0_iter100_max1000.out" \
								 					     "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/13B/fp16/1-gpu-4-batch/nvsmi_numreqsample0_iter100_max500.out" \
								 					     "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/13B/fp16/1-gpu-6-batch/nvsmi_numreqsample0_iter100_max500.out" \
								 --bmark_params          "13 1 1000" \
								 					     "13 1 500" \
													     "13 2 1000" \
													     "13 2 500" \
													     "13 4 1000" \
													     "13 4 500" \
													     "13 6 500" \
								 --plot_power_over_time \
								 --plot_filename 		 "Llama13B_maxlen1000.png" \
								 --plot_sequence_lengths 1000 \
								 --plot_batch_sizes      1 2 4

python gpu_batch_exp_plotting.py --bmark_output_paths    "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/13B/fp16/1-gpu-1-batch/bmark_numreqsample0_iter100_max1000.out" \
								 					     "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/13B/fp16/1-gpu-1-batch/bmark_numreqsample0_iter100_max500.out" \
													     "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/13B/fp16/1-gpu-2-batch/bmark_numreqsample0_iter100_max1000.out" \
								 					     "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/13B/fp16/1-gpu-2-batch/bmark_numreqsample0_iter100_max500.out" \
													     "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/13B/fp16/1-gpu-4-batch/bmark_numreqsample0_iter100_max1000.out" \
								 					     "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/13B/fp16/1-gpu-4-batch/bmark_numreqsample0_iter100_max500.out" \
								 					     "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/13B/fp16/1-gpu-6-batch/bmark_numreqsample0_iter100_max500.out" \
								 --nvsmi_output_paths    "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/13B/fp16/1-gpu-1-batch/nvsmi_numreqsample0_iter100_max1000.out" \
								 					     "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/13B/fp16/1-gpu-1-batch/nvsmi_numreqsample0_iter100_max500.out" \
													     "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/13B/fp16/1-gpu-2-batch/nvsmi_numreqsample0_iter100_max1000.out" \
								 					     "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/13B/fp16/1-gpu-2-batch/nvsmi_numreqsample0_iter100_max500.out" \
													     "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/13B/fp16/1-gpu-4-batch/nvsmi_numreqsample0_iter100_max1000.out" \
								 					     "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/13B/fp16/1-gpu-4-batch/nvsmi_numreqsample0_iter100_max500.out" \
								 					     "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/13B/fp16/1-gpu-6-batch/nvsmi_numreqsample0_iter100_max500.out" \
								 --bmark_params          "13 1 1000" \
								 					     "13 1 500" \
													     "13 2 1000" \
													     "13 2 500" \
													     "13 4 1000" \
													     "13 4 500" \
													     "13 6 500" \
								 --plot_power_over_time \
								 --plot_filename         "Llama13B_maxlen500.png" \
								 --plot_sequence_lengths 500
								 --plot_batch_sizes      1 2 4 6

# TODO: Finish OOM experiments for Llama7B max500
python gpu_batch_exp_plotting.py --bmark_output_paths    "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/7B/fp16/1-gpu-1-batch/bmark_numreqsample0_iter100_max1000.out" \
													     "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/7B/fp16/1-gpu-1-batch/bmark_numreqsample0_iter100_max500.out" \
													     "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/7B/fp16/1-gpu-2-batch/bmark_numreqsample0_iter100_max1000.out" \
													     "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/7B/fp16/1-gpu-2-batch/bmark_numreqsample0_iter100_max500.out" \
													     "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/7B/fp16/1-gpu-4-batch/bmark_numreqsample0_iter100_max1000.out" \
													     "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/7B/fp16/1-gpu-4-batch/bmark_numreqsample0_iter100_max500.out" \
													     "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/7B/fp16/1-gpu-6-batch/bmark_numreqsample0_iter100_max1000.out" \
													     "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/7B/fp16/1-gpu-6-batch/bmark_numreqsample0_iter100_max500.out" \
													     "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/7B/fp16/1-gpu-8-batch/bmark_numreqsample0_iter100_max1000.out" \
													     "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/7B/fp16/1-gpu-8-batch/bmark_numreqsample0_iter100_max500.out" \
													     "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/7B/fp16/1-gpu-10-batch/bmark_numreqsample0_iter100_max1000.out" \
													     "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/7B/fp16/1-gpu-10-batch/bmark_numreqsample0_iter100_max500.out" \
													     "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/7B/fp16/1-gpu-12-batch/bmark_numreqsample0_iter100_max1000.out" \
													     "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/7B/fp16/1-gpu-12-batch/bmark_numreqsample0_iter100_max500.out" \
													     "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/7B/fp16/1-gpu-14-batch/bmark_numreqsample0_iter100_max1000.out" \
													     "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/7B/fp16/1-gpu-14-batch/bmark_numreqsample0_iter100_max500.out" \
													     "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/7B/fp16/1-gpu-16-batch/bmark_numreqsample0_iter100_max500.out" \
													     "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/7B/fp16/1-gpu-18-batch/bmark_numreqsample0_iter100_max500.out" \
													     "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/7B/fp16/1-gpu-20-batch/bmark_numreqsample0_iter100_max500.out" \
													     "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/7B/fp16/1-gpu-22-batch/bmark_numreqsample0_iter100_max500.out" \
													     "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/7B/fp16/1-gpu-24-batch/bmark_numreqsample0_iter100_max500.out" \
													     "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/7B/fp16/1-gpu-26-batch/bmark_numreqsample0_iter100_max500.out" \
							     --nvsmi_output_paths    "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/7B/fp16/1-gpu-1-batch/nvsmi_numreqsample0_iter100_max1000.out" \
													     "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/7B/fp16/1-gpu-1-batch/nvsmi_numreqsample0_iter100_max500.out" \
													     "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/7B/fp16/1-gpu-2-batch/nvsmi_numreqsample0_iter100_max1000.out" \
													     "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/7B/fp16/1-gpu-2-batch/nvsmi_numreqsample0_iter100_max500.out" \
													     "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/7B/fp16/1-gpu-4-batch/nvsmi_numreqsample0_iter100_max1000.out" \
													     "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/7B/fp16/1-gpu-4-batch/nvsmi_numreqsample0_iter100_max500.out" \
													     "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/7B/fp16/1-gpu-6-batch/nvsmi_numreqsample0_iter100_max1000.out" \
													     "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/7B/fp16/1-gpu-6-batch/nvsmi_numreqsample0_iter100_max500.out" \
													     "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/7B/fp16/1-gpu-8-batch/nvsmi_numreqsample0_iter100_max1000.out" \
													     "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/7B/fp16/1-gpu-8-batch/nvsmi_numreqsample0_iter100_max500.out" \
													     "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/7B/fp16/1-gpu-10-batch/nvsmi_numreqsample0_iter100_max1000.out" \
													     "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/7B/fp16/1-gpu-10-batch/nvsmi_numreqsample0_iter100_max500.out" \
													     "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/7B/fp16/1-gpu-12-batch/nvsmi_numreqsample0_iter100_max1000.out" \
													     "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/7B/fp16/1-gpu-12-batch/nvsmi_numreqsample0_iter100_max500.out" \
													     "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/7B/fp16/1-gpu-14-batch/nvsmi_numreqsample0_iter100_max1000.out" \
													     "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/7B/fp16/1-gpu-14-batch/nvsmi_numreqsample0_iter100_max500.out" \
													     "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/7B/fp16/1-gpu-16-batch/nvsmi_numreqsample0_iter100_max500.out" \
													     "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/7B/fp16/1-gpu-18-batch/nvsmi_numreqsample0_iter100_max500.out" \
													     "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/7B/fp16/1-gpu-20-batch/nvsmi_numreqsample0_iter100_max500.out" \
													     "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/7B/fp16/1-gpu-22-batch/nvsmi_numreqsample0_iter100_max500.out" \
													     "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/7B/fp16/1-gpu-24-batch/nvsmi_numreqsample0_iter100_max500.out" \
													     "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/7B/fp16/1-gpu-26-batch/nvsmi_numreqsample0_iter100_max500.out" \
								 --bmark_params          "7 1 1000" \
								 					     "7 1 500" \
													     "7 2 1000" \
													     "7 2 500" \
													     "7 4 1000" \
													     "7 4 500" \
								 					     "7 6 1000" \
								 					     "7 6 500" \
													     "7 8 1000" \
													     "7 8 500" \
													     "7 10 1000" \
													     "7 10 500" \
													     "7 12 1000" \
													     "7 12 500" \
													     "7 14 1000" \
													     "7 14 500" \
													     "7 16 500" \
													     "7 18 500" \
													     "7 20 500" \
													     "7 22 500" \
													     "7 24 500" \
													     "7 26 500" \
							     --plot_power_over_time \
								 --plot_filename         "Llama7B_maxlen1000.png" \
								 --plot_sequence_lengths 1000 \
								 --plot_batch_sizes      1 4 8 12 14

python gpu_batch_exp_plotting.py --bmark_output_paths    "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/7B/fp16/1-gpu-1-batch/bmark_numreqsample0_iter100_max1000.out" \
													     "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/7B/fp16/1-gpu-1-batch/bmark_numreqsample0_iter100_max500.out" \
													     "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/7B/fp16/1-gpu-2-batch/bmark_numreqsample0_iter100_max1000.out" \
													     "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/7B/fp16/1-gpu-2-batch/bmark_numreqsample0_iter100_max500.out" \
													     "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/7B/fp16/1-gpu-4-batch/bmark_numreqsample0_iter100_max1000.out" \
													     "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/7B/fp16/1-gpu-4-batch/bmark_numreqsample0_iter100_max500.out" \
													     "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/7B/fp16/1-gpu-6-batch/bmark_numreqsample0_iter100_max1000.out" \
													     "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/7B/fp16/1-gpu-6-batch/bmark_numreqsample0_iter100_max500.out" \
													     "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/7B/fp16/1-gpu-8-batch/bmark_numreqsample0_iter100_max1000.out" \
													     "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/7B/fp16/1-gpu-8-batch/bmark_numreqsample0_iter100_max500.out" \
													     "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/7B/fp16/1-gpu-10-batch/bmark_numreqsample0_iter100_max1000.out" \
													     "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/7B/fp16/1-gpu-10-batch/bmark_numreqsample0_iter100_max500.out" \
													     "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/7B/fp16/1-gpu-12-batch/bmark_numreqsample0_iter100_max1000.out" \
													     "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/7B/fp16/1-gpu-12-batch/bmark_numreqsample0_iter100_max500.out" \
													     "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/7B/fp16/1-gpu-14-batch/bmark_numreqsample0_iter100_max1000.out" \
													     "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/7B/fp16/1-gpu-14-batch/bmark_numreqsample0_iter100_max500.out" \
													     "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/7B/fp16/1-gpu-16-batch/bmark_numreqsample0_iter100_max500.out" \
													     "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/7B/fp16/1-gpu-18-batch/bmark_numreqsample0_iter100_max500.out" \
													     "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/7B/fp16/1-gpu-20-batch/bmark_numreqsample0_iter100_max500.out" \
													     "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/7B/fp16/1-gpu-22-batch/bmark_numreqsample0_iter100_max500.out" \
													     "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/7B/fp16/1-gpu-24-batch/bmark_numreqsample0_iter100_max500.out" \
													     "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/7B/fp16/1-gpu-26-batch/bmark_numreqsample0_iter100_max500.out" \
							     --nvsmi_output_paths    "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/7B/fp16/1-gpu-1-batch/nvsmi_numreqsample0_iter100_max1000.out" \
													     "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/7B/fp16/1-gpu-1-batch/nvsmi_numreqsample0_iter100_max500.out" \
													     "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/7B/fp16/1-gpu-2-batch/nvsmi_numreqsample0_iter100_max1000.out" \
													     "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/7B/fp16/1-gpu-2-batch/nvsmi_numreqsample0_iter100_max500.out" \
													     "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/7B/fp16/1-gpu-4-batch/nvsmi_numreqsample0_iter100_max1000.out" \
													     "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/7B/fp16/1-gpu-4-batch/nvsmi_numreqsample0_iter100_max500.out" \
													     "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/7B/fp16/1-gpu-6-batch/nvsmi_numreqsample0_iter100_max1000.out" \
													     "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/7B/fp16/1-gpu-6-batch/nvsmi_numreqsample0_iter100_max500.out" \
													     "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/7B/fp16/1-gpu-8-batch/nvsmi_numreqsample0_iter100_max1000.out" \
													     "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/7B/fp16/1-gpu-8-batch/nvsmi_numreqsample0_iter100_max500.out" \
													     "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/7B/fp16/1-gpu-10-batch/nvsmi_numreqsample0_iter100_max1000.out" \
													     "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/7B/fp16/1-gpu-10-batch/nvsmi_numreqsample0_iter100_max500.out" \
													     "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/7B/fp16/1-gpu-12-batch/nvsmi_numreqsample0_iter100_max1000.out" \
													     "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/7B/fp16/1-gpu-12-batch/nvsmi_numreqsample0_iter100_max500.out" \
													     "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/7B/fp16/1-gpu-14-batch/nvsmi_numreqsample0_iter100_max1000.out" \
													     "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/7B/fp16/1-gpu-14-batch/nvsmi_numreqsample0_iter100_max500.out" \
													     "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/7B/fp16/1-gpu-16-batch/nvsmi_numreqsample0_iter100_max500.out" \
													     "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/7B/fp16/1-gpu-18-batch/nvsmi_numreqsample0_iter100_max500.out" \
													     "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/7B/fp16/1-gpu-20-batch/nvsmi_numreqsample0_iter100_max500.out" \
													     "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/7B/fp16/1-gpu-22-batch/nvsmi_numreqsample0_iter100_max500.out" \
													     "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/7B/fp16/1-gpu-24-batch/nvsmi_numreqsample0_iter100_max500.out" \
													     "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/7B/fp16/1-gpu-26-batch/nvsmi_numreqsample0_iter100_max500.out" \
								 --bmark_params          "7 1 1000" \
								 					     "7 1 500" \
													     "7 2 1000" \
													     "7 2 500" \
													     "7 4 1000" \
													     "7 4 500" \
								 					     "7 6 1000" \
								 					     "7 6 500" \
													     "7 8 1000" \
													     "7 8 500" \
													     "7 10 1000" \
													     "7 10 500" \
													     "7 12 1000" \
													     "7 12 500" \
													     "7 14 1000" \
													     "7 14 500" \
													     "7 16 500" \
													     "7 18 500" \
													     "7 20 500" \
													     "7 22 500" \
													     "7 24 500" \
													     "7 26 500" \
							     --plot_power_over_time \
								 --plot_filename         "Llama7B_maxlen500.png" \
								 --plot_sequence_lengths 500 \
								 --plot_batch_sizes      1 8 16 24 26

python gpu_batch_exp_plotting.py --bmark_output_paths         "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/13B/fp16/1-gpu-1-batch/bmark_numreqsample0_iter100_max1000.out" \
								 					          "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/13B/fp16/1-gpu-1-batch/bmark_numreqsample0_iter100_max500.out" \
													          "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/13B/fp16/1-gpu-2-batch/bmark_numreqsample0_iter100_max1000.out" \
								 					          "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/13B/fp16/1-gpu-2-batch/bmark_numreqsample0_iter100_max500.out" \
													          "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/13B/fp16/1-gpu-4-batch/bmark_numreqsample0_iter100_max1000.out" \
								 					          "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/13B/fp16/1-gpu-4-batch/bmark_numreqsample0_iter100_max500.out" \
								 					          "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/13B/fp16/1-gpu-6-batch/bmark_numreqsample0_iter100_max500.out" \
															  "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/7B/fp16/1-gpu-1-batch/bmark_numreqsample0_iter100_max1000.out" \
													          "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/7B/fp16/1-gpu-1-batch/bmark_numreqsample0_iter100_max500.out" \
													          "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/7B/fp16/1-gpu-2-batch/bmark_numreqsample0_iter100_max1000.out" \
													          "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/7B/fp16/1-gpu-2-batch/bmark_numreqsample0_iter100_max500.out" \
													          "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/7B/fp16/1-gpu-4-batch/bmark_numreqsample0_iter100_max1000.out" \
													          "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/7B/fp16/1-gpu-4-batch/bmark_numreqsample0_iter100_max500.out" \
													          "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/7B/fp16/1-gpu-6-batch/bmark_numreqsample0_iter100_max1000.out" \
													          "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/7B/fp16/1-gpu-6-batch/bmark_numreqsample0_iter100_max500.out" \
													          "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/7B/fp16/1-gpu-8-batch/bmark_numreqsample0_iter100_max1000.out" \
													          "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/7B/fp16/1-gpu-8-batch/bmark_numreqsample0_iter100_max500.out" \
													          "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/7B/fp16/1-gpu-10-batch/bmark_numreqsample0_iter100_max1000.out" \
													          "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/7B/fp16/1-gpu-10-batch/bmark_numreqsample0_iter100_max500.out" \
													          "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/7B/fp16/1-gpu-12-batch/bmark_numreqsample0_iter100_max1000.out" \
													          "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/7B/fp16/1-gpu-12-batch/bmark_numreqsample0_iter100_max500.out" \
													          "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/7B/fp16/1-gpu-14-batch/bmark_numreqsample0_iter100_max1000.out" \
													          "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/7B/fp16/1-gpu-14-batch/bmark_numreqsample0_iter100_max500.out" \
													          "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/7B/fp16/1-gpu-16-batch/bmark_numreqsample0_iter100_max500.out" \
													          "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/7B/fp16/1-gpu-18-batch/bmark_numreqsample0_iter100_max500.out" \
													          "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/7B/fp16/1-gpu-20-batch/bmark_numreqsample0_iter100_max500.out" \
													          "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/7B/fp16/1-gpu-22-batch/bmark_numreqsample0_iter100_max500.out" \
													          "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/7B/fp16/1-gpu-24-batch/bmark_numreqsample0_iter100_max500.out" \
													          "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/7B/fp16/1-gpu-26-batch/bmark_numreqsample0_iter100_max500.out" \
								 --nvsmi_output_paths         "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/13B/fp16/1-gpu-1-batch/nvsmi_numreqsample0_iter100_max1000.out" \
								 					          "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/13B/fp16/1-gpu-1-batch/nvsmi_numreqsample0_iter100_max500.out" \
													          "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/13B/fp16/1-gpu-2-batch/nvsmi_numreqsample0_iter100_max1000.out" \
								 					          "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/13B/fp16/1-gpu-2-batch/nvsmi_numreqsample0_iter100_max500.out" \
													          "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/13B/fp16/1-gpu-4-batch/nvsmi_numreqsample0_iter100_max1000.out" \
								 					          "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/13B/fp16/1-gpu-4-batch/nvsmi_numreqsample0_iter100_max500.out" \
								 					          "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/13B/fp16/1-gpu-6-batch/nvsmi_numreqsample0_iter100_max500.out" \
							       							  "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/7B/fp16/1-gpu-1-batch/nvsmi_numreqsample0_iter100_max1000.out" \
													          "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/7B/fp16/1-gpu-1-batch/nvsmi_numreqsample0_iter100_max500.out" \
													          "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/7B/fp16/1-gpu-2-batch/nvsmi_numreqsample0_iter100_max1000.out" \
													          "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/7B/fp16/1-gpu-2-batch/nvsmi_numreqsample0_iter100_max500.out" \
													          "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/7B/fp16/1-gpu-4-batch/nvsmi_numreqsample0_iter100_max1000.out" \
													          "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/7B/fp16/1-gpu-4-batch/nvsmi_numreqsample0_iter100_max500.out" \
													          "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/7B/fp16/1-gpu-6-batch/nvsmi_numreqsample0_iter100_max1000.out" \
													          "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/7B/fp16/1-gpu-6-batch/nvsmi_numreqsample0_iter100_max500.out" \
													          "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/7B/fp16/1-gpu-8-batch/nvsmi_numreqsample0_iter100_max1000.out" \
													          "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/7B/fp16/1-gpu-8-batch/nvsmi_numreqsample0_iter100_max500.out" \
													          "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/7B/fp16/1-gpu-10-batch/nvsmi_numreqsample0_iter100_max1000.out" \
													          "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/7B/fp16/1-gpu-10-batch/nvsmi_numreqsample0_iter100_max500.out" \
													          "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/7B/fp16/1-gpu-12-batch/nvsmi_numreqsample0_iter100_max1000.out" \
													          "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/7B/fp16/1-gpu-12-batch/nvsmi_numreqsample0_iter100_max500.out" \
													          "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/7B/fp16/1-gpu-14-batch/nvsmi_numreqsample0_iter100_max1000.out" \
													          "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/7B/fp16/1-gpu-14-batch/nvsmi_numreqsample0_iter100_max500.out" \
													          "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/7B/fp16/1-gpu-16-batch/nvsmi_numreqsample0_iter100_max500.out" \
													          "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/7B/fp16/1-gpu-18-batch/nvsmi_numreqsample0_iter100_max500.out" \
													          "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/7B/fp16/1-gpu-20-batch/nvsmi_numreqsample0_iter100_max500.out" \
													          "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/7B/fp16/1-gpu-22-batch/nvsmi_numreqsample0_iter100_max500.out" \
													          "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/7B/fp16/1-gpu-24-batch/nvsmi_numreqsample0_iter100_max500.out" \
													          "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/7B/fp16/1-gpu-26-batch/nvsmi_numreqsample0_iter100_max500.out" \
								 --bmark_params               "13 1 1000" \
								 					          "13 1 500" \
													          "13 2 1000" \
													          "13 2 500" \
													          "13 4 1000" \
													          "13 4 500" \
													          "13 6 500" \
								                              "7 1 1000" \
								 					          "7 1 500" \
													          "7 2 1000" \
													          "7 2 500" \
													          "7 4 1000" \
													          "7 4 500" \
								 					          "7 6 1000" \
								 					          "7 6 500" \
													          "7 8 1000" \
													          "7 8 500" \
													          "7 10 1000" \
													          "7 10 500" \
													          "7 12 1000" \
													          "7 12 500" \
													          "7 14 1000" \
													          "7 14 500" \
													          "7 16 500" \
													          "7 18 500" \
													          "7 20 500" \
													          "7 22 500" \
													          "7 24 500" \
													          "7 26 500" \
								 --bmark_param_groups         "13 X 1000" \
								 							  "13 X 500" \
															  "7 X 1000" \
															  "7 X 500" \
								 --plot_average_batch_latency \
								 --plot_filename 		      "Llama_avg_batch_latency.png" \
								 --plot_sequence_lengths      1000 500 \
								 --plot_batch_sizes           1 2 4 6 8 10 12 14 16 18 20 22 24 26


python gpu_batch_exp_plotting.py --bmark_output_paths         "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/13B/fp16/1-gpu-1-batch/bmark_numreqsample0_iter100_max1000.out" \
													          "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/13B/fp16/1-gpu-2-batch/bmark_numreqsample0_iter100_max1000.out" \
													          "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/13B/fp16/1-gpu-4-batch/bmark_numreqsample0_iter100_max1000.out" \
															  "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/7B/fp16/1-gpu-1-batch/bmark_numreqsample0_iter100_max1000.out" \
													          "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/7B/fp16/1-gpu-2-batch/bmark_numreqsample0_iter100_max1000.out" \
													          "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/7B/fp16/1-gpu-4-batch/bmark_numreqsample0_iter100_max1000.out" \
													          "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/7B/fp16/1-gpu-6-batch/bmark_numreqsample0_iter100_max1000.out" \
													          "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/7B/fp16/1-gpu-8-batch/bmark_numreqsample0_iter100_max1000.out" \
													          "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/7B/fp16/1-gpu-10-batch/bmark_numreqsample0_iter100_max1000.out" \
													          "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/7B/fp16/1-gpu-12-batch/bmark_numreqsample0_iter100_max1000.out" \
													          "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/7B/fp16/1-gpu-14-batch/bmark_numreqsample0_iter100_max1000.out" \
								 --nvsmi_output_paths         "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/13B/fp16/1-gpu-1-batch/nvsmi_numreqsample0_iter100_max1000.out" \
													          "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/13B/fp16/1-gpu-2-batch/nvsmi_numreqsample0_iter100_max1000.out" \
													          "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/13B/fp16/1-gpu-4-batch/nvsmi_numreqsample0_iter100_max1000.out" \
							       							  "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/7B/fp16/1-gpu-1-batch/nvsmi_numreqsample0_iter100_max1000.out" \
													          "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/7B/fp16/1-gpu-2-batch/nvsmi_numreqsample0_iter100_max1000.out" \
													          "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/7B/fp16/1-gpu-4-batch/nvsmi_numreqsample0_iter100_max1000.out" \
													          "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/7B/fp16/1-gpu-6-batch/nvsmi_numreqsample0_iter100_max1000.out" \
													          "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/7B/fp16/1-gpu-8-batch/nvsmi_numreqsample0_iter100_max1000.out" \
													          "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/7B/fp16/1-gpu-10-batch/nvsmi_numreqsample0_iter100_max1000.out" \
													          "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/7B/fp16/1-gpu-12-batch/nvsmi_numreqsample0_iter100_max1000.out" \
													          "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/7B/fp16/1-gpu-14-batch/nvsmi_numreqsample0_iter100_max1000.out" \
								 --bmark_params               "13 1 1000" \
													          "13 2 1000" \
													          "13 4 1000" \
								                              "7 1 1000" \
													          "7 2 1000" \
													          "7 4 1000" \
								 					          "7 6 1000" \
													          "7 8 1000" \
													          "7 10 1000" \
													          "7 12 1000" \
													          "7 14 1000" \
								 --plot_normalized_token_latency \
								 --plot_filename 		      "Llama_normalized_token_latency.png" \
								 --plot_sequence_lengths      1000 \
								 --plot_batch_sizes           1 2 4 6 8 10 12 14 16 18 20 22 24 26



# converting all max 1000 sequence outputs to new output format for 13B nowq
python convert_output_format.py --bmark_output_paths "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/13B/fp16/1-gpu-1-batch/bmark_numreqsample0_iter100_max1000_a10040gb.out" \
													 "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/13B/fp16/1-gpu-1-batch/bmark_numreqsample0_iter100_max1000_v10032gb.out" \
													 "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/13B/fp16/1-gpu-2-batch/bmark_numreqsample0_iter100_max1000_a10040gb.out" \
													 "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/13B/fp16/1-gpu-2-batch/bmark_numreqsample0_iter100_max1000_v10032gb.out" \
													 "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/13B/fp16/1-gpu-4-batch/bmark_numreqsample0_iter100_max1000_a10040gb.out" \
													 "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/13B/fp16/1-gpu-4-batch/bmark_numreqsample0_iter100_max1000_v10032gb.out" \
													 "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/13B/fp16/1-gpu-6-batch/bmark_numreqsample0_iter100_max1000_a10040gb.out" \
													 "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/13B/fp16/1-gpu-8-batch/bmark_numreqsample0_iter100_max1000_a10040gb.out"

python convert_output_format.py --nvsmi_output_paths "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/13B/fp16/1-gpu-1-batch/nvsmi_numreqsample0_iter100_max1000_a10040gb.out" \
													 "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/13B/fp16/1-gpu-1-batch/nvsmi_numreqsample0_iter100_max1000_v10032gb.out" \
													 "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/13B/fp16/1-gpu-2-batch/nvsmi_numreqsample0_iter100_max1000_a10040gb.out" \
													 "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/13B/fp16/1-gpu-2-batch/nvsmi_numreqsample0_iter100_max1000_v10032gb.out" \
													 "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/13B/fp16/1-gpu-4-batch/nvsmi_numreqsample0_iter100_max1000_a10040gb.out" \
													 "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/13B/fp16/1-gpu-4-batch/nvsmi_numreqsample0_iter100_max1000_v10032gb.out" \
													 "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/13B/fp16/1-gpu-6-batch/nvsmi_numreqsample0_iter100_max1000_a10040gb.out" \
													 "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/13B/fp16/1-gpu-8-batch/nvsmi_numreqsample0_iter100_max1000_a10040gb.out" \
								--num_gpus			 2

# converting all max 1000 sequence outputs to new output format for 7B nowq
python convert_output_format.py --bmark_output_paths "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/7B/fp16/1-gpu-1-batch/bmark_numreqsample0_iter100_max1000_a10040gb.out" \
													 "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/7B/fp16/1-gpu-1-batch/bmark_numreqsample0_iter100_max1000_v10032gb.out" \
													 "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/7B/fp16/1-gpu-2-batch/bmark_numreqsample0_iter100_max1000_v10032gb.out" \
													 "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/7B/fp16/1-gpu-4-batch/bmark_numreqsample0_iter100_max1000_a10040gb.out" \
													 "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/7B/fp16/1-gpu-4-batch/bmark_numreqsample0_iter100_max1000_v10032gb.out" \
													 "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/7B/fp16/1-gpu-6-batch/bmark_numreqsample0_iter100_max1000_v10032gb.out" \
													 "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/7B/fp16/1-gpu-8-batch/bmark_numreqsample0_iter100_max1000_a10040gb.out" \
													 "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/7B/fp16/1-gpu-8-batch/bmark_numreqsample0_iter100_max1000_v10032gb.out" \
													 "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/7B/fp16/1-gpu-10-batch/bmark_numreqsample0_iter100_max1000_v10032gb.out" \
													 "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/7B/fp16/1-gpu-12-batch/bmark_numreqsample0_iter100_max1000_a10040gb.out" \
													 "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/7B/fp16/1-gpu-12-batch/bmark_numreqsample0_iter100_max1000_v10032gb.out" \
													 "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/7B/fp16/1-gpu-14-batch/bmark_numreqsample0_iter100_max1000_v10032gb.out" \
													 "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/7B/fp16/1-gpu-16-batch/bmark_numreqsample0_iter100_max1000_a10040gb.out" \
													 "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/7B/fp16/1-gpu-20-batch/bmark_numreqsample0_iter100_max1000_a10040gb.out" \
													 "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/7B/fp16/1-gpu-22-batch/bmark_numreqsample0_iter100_max1000_a10040gb.out"

# testing
python convert_output_format.py --bmark_output_paths "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/13B/fp16/1-gpu-2-batch/bmark_numreqsample0_iter100_max1000_v10032gb.out"

# reformatting all max 1000 sequence nvsmi outputs for new output format for 7b nowq
python convert_output_format.py --nvsmi_output_paths "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/7B/fp16/1-gpu-1-batch/nvsmi_numreqsample0_iter100_max1000_a10040gb.out" \
													 "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/7B/fp16/1-gpu-1-batch/nvsmi_numreqsample0_iter100_max1000_v10032gb.out" \
													 "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/7B/fp16/1-gpu-2-batch/nvsmi_numreqsample0_iter100_max1000_v10032gb.out" \
													 "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/7B/fp16/1-gpu-4-batch/nvsmi_numreqsample0_iter100_max1000_a10040gb.out" \
													 "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/7B/fp16/1-gpu-4-batch/nvsmi_numreqsample0_iter100_max1000_v10032gb.out" \
													 "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/7B/fp16/1-gpu-6-batch/nvsmi_numreqsample0_iter100_max1000_v10032gb.out" \
													 "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/7B/fp16/1-gpu-8-batch/nvsmi_numreqsample0_iter100_max1000_a10040gb.out" \
													 "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/7B/fp16/1-gpu-8-batch/nvsmi_numreqsample0_iter100_max1000_v10032gb.out" \
													 "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/7B/fp16/1-gpu-10-batch/nvsmi_numreqsample0_iter100_max1000_v10032gb.out" \
													 "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/7B/fp16/1-gpu-12-batch/nvsmi_numreqsample0_iter100_max1000_a10040gb.out" \
													 "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/7B/fp16/1-gpu-12-batch/nvsmi_numreqsample0_iter100_max1000_v10032gb.out" \
													 "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/7B/fp16/1-gpu-14-batch/nvsmi_numreqsample0_iter100_max1000_v10032gb.out" \
													 "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/7B/fp16/1-gpu-16-batch/nvsmi_numreqsample0_iter100_max1000_a10040gb.out" \
													 "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/7B/fp16/1-gpu-20-batch/nvsmi_numreqsample0_iter100_max1000_a10040gb.out" \
													 "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/7B/fp16/1-gpu-22-batch/nvsmi_numreqsample0_iter100_max1000_a10040gb.out" \
                                --num_gpus 2

# (7B + 13B) x (40GB A100 + 32GB V100) nowq normalized token latency plot (everything combined w/o quantization)
python gpu_batch_exp_plotting.py --bmark_output_paths "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/7B/fp16/1-gpu-1-batch/bmark_numreqsample0_iter100_max1000_a10040gb.out" \
													  "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/7B/fp16/1-gpu-4-batch/bmark_numreqsample0_iter100_max1000_a10040gb.out" \
													  "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/7B/fp16/1-gpu-8-batch/bmark_numreqsample0_iter100_max1000_a10040gb.out" \
													  "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/7B/fp16/1-gpu-12-batch/bmark_numreqsample0_iter100_max1000_a10040gb.out" \
													  "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/7B/fp16/1-gpu-16-batch/bmark_numreqsample0_iter100_max1000_a10040gb.out" \
													  "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/7B/fp16/1-gpu-20-batch/bmark_numreqsample0_iter100_max1000_a10040gb.out" \
													  "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/7B/fp16/1-gpu-22-batch/bmark_numreqsample0_iter100_max1000_a10040gb.out" \
													  "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/7B/fp16/1-gpu-1-batch/bmark_numreqsample0_iter100_max1000_v10032gb.out" \
													  "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/7B/fp16/1-gpu-4-batch/bmark_numreqsample0_iter100_max1000_v10032gb.out" \
													  "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/7B/fp16/1-gpu-8-batch/bmark_numreqsample0_iter100_max1000_v10032gb.out" \
													  "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/7B/fp16/1-gpu-12-batch/bmark_numreqsample0_iter100_max1000_v10032gb.out" \
													  "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/7B/fp16/1-gpu-14-batch/bmark_numreqsample0_iter100_max1000_v10032gb.out" \
													  "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/13B/fp16/1-gpu-1-batch/bmark_numreqsample0_iter100_max1000_a10040gb.out" \
													  "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/13B/fp16/1-gpu-2-batch/bmark_numreqsample0_iter100_max1000_a10040gb.out" \
													  "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/13B/fp16/1-gpu-4-batch/bmark_numreqsample0_iter100_max1000_a10040gb.out" \
													  "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/13B/fp16/1-gpu-6-batch/bmark_numreqsample0_iter100_max1000_a10040gb.out" \
													  "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/13B/fp16/1-gpu-8-batch/bmark_numreqsample0_iter100_max1000_a10040gb.out" \
													  "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/13B/fp16/1-gpu-1-batch/bmark_numreqsample0_iter100_max1000_v10032gb.out" \
													  "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/13B/fp16/1-gpu-2-batch/bmark_numreqsample0_iter100_max1000_v10032gb.out" \
													  "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/13B/fp16/1-gpu-4-batch/bmark_numreqsample0_iter100_max1000_v10032gb.out" \
								 --nvsmi_output_paths "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/7B/fp16/1-gpu-1-batch/nvsmi_numreqsample0_iter100_max1000_a10040gb.out" \
													  "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/7B/fp16/1-gpu-4-batch/nvsmi_numreqsample0_iter100_max1000_a10040gb.out" \
													  "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/7B/fp16/1-gpu-8-batch/nvsmi_numreqsample0_iter100_max1000_a10040gb.out" \
													  "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/7B/fp16/1-gpu-12-batch/nvsmi_numreqsample0_iter100_max1000_a10040gb.out" \
													  "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/7B/fp16/1-gpu-16-batch/nvsmi_numreqsample0_iter100_max1000_a10040gb.out" \
													  "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/7B/fp16/1-gpu-20-batch/nvsmi_numreqsample0_iter100_max1000_a10040gb.out" \
													  "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/7B/fp16/1-gpu-22-batch/nvsmi_numreqsample0_iter100_max1000_a10040gb.out" \
													  "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/7B/fp16/1-gpu-1-batch/nvsmi_numreqsample0_iter100_max1000_v10032gb.out" \
													  "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/7B/fp16/1-gpu-4-batch/nvsmi_numreqsample0_iter100_max1000_v10032gb.out" \
													  "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/7B/fp16/1-gpu-8-batch/nvsmi_numreqsample0_iter100_max1000_v10032gb.out" \
													  "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/7B/fp16/1-gpu-12-batch/nvsmi_numreqsample0_iter100_max1000_v10032gb.out" \
													  "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/7B/fp16/1-gpu-14-batch/nvsmi_numreqsample0_iter100_max1000_v10032gb.out" \
													  "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/13B/fp16/1-gpu-1-batch/nvsmi_numreqsample0_iter100_max1000_a10040gb.out" \
													  "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/13B/fp16/1-gpu-2-batch/nvsmi_numreqsample0_iter100_max1000_a10040gb.out" \
													  "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/13B/fp16/1-gpu-4-batch/nvsmi_numreqsample0_iter100_max1000_a10040gb.out" \
													  "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/13B/fp16/1-gpu-6-batch/nvsmi_numreqsample0_iter100_max1000_a10040gb.out" \
													  "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/13B/fp16/1-gpu-8-batch/nvsmi_numreqsample0_iter100_max1000_a10040gb.out" \
													  "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/13B/fp16/1-gpu-1-batch/nvsmi_numreqsample0_iter100_max1000_v10032gb.out" \
													  "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/13B/fp16/1-gpu-2-batch/nvsmi_numreqsample0_iter100_max1000_v10032gb.out" \
													  "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/13B/fp16/1-gpu-4-batch/nvsmi_numreqsample0_iter100_max1000_v10032gb.out" \
								 --bmark_params       "7 1 1000 a10040gb" \
								 					  "7 4 1000 a10040gb" \
													  "7 8 1000 a10040gb" \
													  "7 12 1000 a10040gb" \
													  "7 16 1000 a10040gb" \
													  "7 20 1000 a10040gb" \
													  "7 22 1000 a10040gb" \
													  "7 1 1000 v10032gb" \
													  "7 4 1000 v10032gb" \
													  "7 8 1000 v10032gb" \
													  "7 12 1000 v10032gb" \
													  "7 14 1000 v10032gb" \
													  "13 1 1000 a10040gb" \
													  "13 2 1000 a10040gb" \
													  "13 4 1000 a10040gb" \
													  "13 6 1000 a10040gb" \
													  "13 8 1000 a10040gb" \
													  "13 1 1000 v10032gb" \
													  "13 2 1000 v10032gb" \
													  "13 4 1000 v10032gb" \
								 --bmark_param_groups "7 X 1000 a10040gb" \
								 					  "7 X 1000 v10032gb" \
													  "13 X 1000 a10040gb" \
													  "13 X 1000 v10032gb" \
								 --excluded_tokens    32000 2 \
								 --plot_normalized_token_latency \
								 --plot_filename      "llama2_normalized_token_latency.png"



python gpu_batch_exp_plotting.py --bmark_output_paths "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/7B/fp16/1-gpu-1-batch/bmark_numreqsample0_iter100_max1000_a10040gb.out" \
													  "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/7B/fp16/1-gpu-4-batch/bmark_numreqsample0_iter100_max1000_a10040gb.out" \
													  "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/7B/fp16/1-gpu-8-batch/bmark_numreqsample0_iter100_max1000_a10040gb.out" \
													  "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/7B/fp16/1-gpu-12-batch/bmark_numreqsample0_iter100_max1000_a10040gb.out" \
													  "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/7B/fp16/1-gpu-16-batch/bmark_numreqsample0_iter100_max1000_a10040gb.out" \
													  "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/7B/fp16/1-gpu-20-batch/bmark_numreqsample0_iter100_max1000_a10040gb.out" \
													  "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/7B/fp16/1-gpu-22-batch/bmark_numreqsample0_iter100_max1000_a10040gb.out" \
								 --nvsmi_output_paths "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/7B/fp16/1-gpu-1-batch/nvsmi_numreqsample0_iter100_max1000_a10040gb.out" \
													  "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/7B/fp16/1-gpu-4-batch/nvsmi_numreqsample0_iter100_max1000_a10040gb.out" \
													  "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/7B/fp16/1-gpu-8-batch/nvsmi_numreqsample0_iter100_max1000_a10040gb.out" \
													  "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/7B/fp16/1-gpu-12-batch/nvsmi_numreqsample0_iter100_max1000_a10040gb.out" \
													  "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/7B/fp16/1-gpu-16-batch/nvsmi_numreqsample0_iter100_max1000_a10040gb.out" \
													  "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/7B/fp16/1-gpu-20-batch/nvsmi_numreqsample0_iter100_max1000_a10040gb.out" \
													  "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/7B/fp16/1-gpu-22-batch/nvsmi_numreqsample0_iter100_max1000_a10040gb.out" \
								 --bmark_params       "7 1 1000 a10040gb" \
								 					  "7 4 1000 a10040gb" \
													  "7 8 1000 a10040gb" \
													  "7 12 1000 a10040gb" \
													  "7 16 1000 a10040gb" \
													  "7 20 1000 a10040gb" \
													  "7 22 1000 a10040gb" \
								 --bmark_param_groups "7 X 1000 a10040gb" \
								 --excluded_tokens    32000 2 \
								 --plot_normalized_token_latency \
								 --plot_filename      "llama2_7b_utilization_a10040gb.png"



python gpu_batch_exp_plotting.py --bmark_output_paths "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/7B/fp16/1-gpu-1-batch/bmark_numreqsample0_iter100_max1000_v10032gb.out" \
													  "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/7B/fp16/1-gpu-4-batch/bmark_numreqsample0_iter100_max1000_v10032gb.out" \
													  "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/7B/fp16/1-gpu-8-batch/bmark_numreqsample0_iter100_max1000_v10032gb.out" \
													  "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/7B/fp16/1-gpu-12-batch/bmark_numreqsample0_iter100_max1000_v10032gb.out" \
													  "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/7B/fp16/1-gpu-14-batch/bmark_numreqsample0_iter100_max1000_v10032gb.out" \
								 --nvsmi_output_paths "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/7B/fp16/1-gpu-1-batch/nvsmi_numreqsample0_iter100_max1000_v10032gb.out" \
													  "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/7B/fp16/1-gpu-4-batch/nvsmi_numreqsample0_iter100_max1000_v10032gb.out" \
													  "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/7B/fp16/1-gpu-8-batch/nvsmi_numreqsample0_iter100_max1000_v10032gb.out" \
													  "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/7B/fp16/1-gpu-12-batch/nvsmi_numreqsample0_iter100_max1000_v10032gb.out" \
													  "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/7B/fp16/1-gpu-14-batch/nvsmi_numreqsample0_iter100_max1000_v10032gb.out" \
								 --bmark_params       "7 1 1000 v10032gb" \
													  "7 4 1000 v10032gb" \
													  "7 8 1000 v10032gb" \
													  "7 12 1000 v10032gb" \
													  "7 14 1000 v10032gb" \
								 --bmark_param_groups "7 X 1000 v10032gb" \
								 --excluded_tokens    32000 2 \
								 --plot_normalized_token_latency \
								 --plot_filename      "llama2_normalized_token_latency.png"

python convert_output_format.py --bmark_output_paths "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/7B/fp16/1-gpu-1-batch/bmark_numreqsample0_iter100_max500_v10032gb.out"
python convert_output_format.py --nvsmi_output_paths "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/7B/fp16/1-gpu-1-batch/nvsmi_numreqsample0_iter100_max500_v10032gb.out" \
                                --num_gpus 2

python gpu_batch_exp_plotting.py --bmark_output_paths "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/7B/fp16/1-gpu-1-batch/bmark_numreqsample0_iter100_max1000_v10032gb.out" \
													  "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/7B/fp16/1-gpu-1-batch/bmark_numreqsample0_iter100_max1000_a10040gb.out" \
													  "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/7B/fp16/1-gpu-1-batch/bmark_numreqsample0_iter100_max500_v10032gb.out" \
													  "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/7B/fp16_wq4/1-gpu-1-batch/bmark_numreqsample0_iter100_max1000_a10040gb.out" \
													  "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/7B/fp16_wq4/1-gpu-1-batch/bmark_numreqsample0_iter100_max1000_v10032gb.out" \
								 --nvsmi_output_paths "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/7B/fp16/1-gpu-1-batch/nvsmi_numreqsample0_iter100_max1000_v10032gb.out" \
													  "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/7B/fp16/1-gpu-1-batch/nvsmi_numreqsample0_iter100_max1000_a10040gb.out" \
													  "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/7B/fp16/1-gpu-1-batch/nvsmi_numreqsample0_iter100_max500_v10032gb.out" \
													  "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/7B/fp16_wq4/1-gpu-1-batch/nvsmi_numreqsample0_iter100_max1000_a10040gb.out" \
													  "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/7B/fp16_wq4/1-gpu-1-batch/nvsmi_numreqsample0_iter100_max1000_v10032gb.out" \
								 --bmark_params 	  "7 1 1000 v10032gb" \
								                      "7 1 1000 a10040gb" \
													  "7 1 500 v10032gb" \
													  "7 1 1000 a10040gb" \
													  "7 1 1000 v10032gb" \
							     --plot_power_over_time \
								 --gpu_idx 0 \
								 --plot_filename      "Llama_7B_Power_Usage.png"

python fix_nvsmi_output_timing.py --bmark_output_path  "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/7B/fp16/1-gpu-1-batch/bmark_numreqsample0_iter100_max1000_v10032gb.out" \
								  --nvsmi_output_path  "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/7B/fp16/1-gpu-1-batch/nvsmi_numreqsample0_iter100_max1000_v10032gb.out" \
								  --gpu_idx            0 \
								  --nvsmi_start_line   104 \
								  --nvsmi_end_line     470

python fix_nvsmi_output_timing.py --bmark_output_path  "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/7B/fp16/1-gpu-1-batch/bmark_numreqsample0_iter100_max1000_a10040gb.out" \
								  --nvsmi_output_path  "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/7B/fp16/1-gpu-1-batch/nvsmi_numreqsample0_iter100_max1000_a10040gb.out" \
								  --gpu_idx            0 \
								  --nvsmi_start_line   37 \
								  --nvsmi_end_line     145

# a100 llama unquantized
python gpu_batch_exp_plotting.py --bmark_output_paths "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/7B/fp16/1-gpu-1-batch/bmark_numreqsample0_iter100_max1000_a10040gb.out" \
													  "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/7B/fp16/1-gpu-4-batch/bmark_numreqsample0_iter100_max1000_a10040gb.out" \
													  "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/7B/fp16/1-gpu-8-batch/bmark_numreqsample0_iter100_max1000_a10040gb.out" \
													  "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/7B/fp16/1-gpu-12-batch/bmark_numreqsample0_iter100_max1000_a10040gb.out" \
													  "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/7B/fp16/1-gpu-16-batch/bmark_numreqsample0_iter100_max1000_a10040gb.out" \
													  "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/7B/fp16/1-gpu-20-batch/bmark_numreqsample0_iter100_max1000_a10040gb.out" \
													  "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/7B/fp16/1-gpu-22-batch/bmark_numreqsample0_iter100_max1000_a10040gb.out" \
													  "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/13B/fp16/1-gpu-1-batch/bmark_numreqsample0_iter100_max1000_a10040gb.out" \
													  "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/13B/fp16/1-gpu-2-batch/bmark_numreqsample0_iter100_max1000_a10040gb.out" \
													  "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/13B/fp16/1-gpu-4-batch/bmark_numreqsample0_iter100_max1000_a10040gb.out" \
													  "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/13B/fp16/1-gpu-6-batch/bmark_numreqsample0_iter100_max1000_a10040gb.out" \
													  "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/13B/fp16/1-gpu-8-batch/bmark_numreqsample0_iter100_max1000_a10040gb.out" \
								 --nvsmi_output_paths "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/7B/fp16/1-gpu-1-batch/nvsmi_numreqsample0_iter100_max1000_a10040gb.out" \
													  "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/7B/fp16/1-gpu-4-batch/nvsmi_numreqsample0_iter100_max1000_a10040gb.out" \
													  "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/7B/fp16/1-gpu-8-batch/nvsmi_numreqsample0_iter100_max1000_a10040gb.out" \
													  "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/7B/fp16/1-gpu-12-batch/nvsmi_numreqsample0_iter100_max1000_a10040gb.out" \
													  "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/7B/fp16/1-gpu-16-batch/nvsmi_numreqsample0_iter100_max1000_a10040gb.out" \
													  "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/7B/fp16/1-gpu-20-batch/nvsmi_numreqsample0_iter100_max1000_a10040gb.out" \
													  "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/7B/fp16/1-gpu-22-batch/nvsmi_numreqsample0_iter100_max1000_a10040gb.out" \
													  "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/13B/fp16/1-gpu-1-batch/nvsmi_numreqsample0_iter100_max1000_a10040gb.out" \
													  "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/13B/fp16/1-gpu-2-batch/nvsmi_numreqsample0_iter100_max1000_a10040gb.out" \
													  "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/13B/fp16/1-gpu-4-batch/nvsmi_numreqsample0_iter100_max1000_a10040gb.out" \
													  "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/13B/fp16/1-gpu-6-batch/nvsmi_numreqsample0_iter100_max1000_a10040gb.out" \
													  "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/13B/fp16/1-gpu-8-batch/nvsmi_numreqsample0_iter100_max1000_a10040gb.out" \
								 --bmark_params       "7B 1 1000 a10040gb" \
								                      "7B 4 1000 a10040gb" \
													  "7B 8 1000 a10040gb" \
													  "7B 12 1000 a10040gb" \
													  "7B 16 1000 a10040gb" \
													  "7B 20 1000 a10040gb" \
													  "7B 22 1000 a10040gb" \
													  "13B 1 1000 a10040gb" \
													  "13B 2 1000 a10040gb" \
													  "13B 4 1000 a10040gb" \
													  "13B 6 1000 a10040gb" \
													  "13B 8 1000 a10040gb" \
								 --bmark_param_groups "7B X 1000 a10040gb" \
								 					  "13B X 1000 a10040gb" \
								 --plot_filename      "throughput_vs_latency_a10040gb.png" \
								 --plot_name          "A100-SXM4-40GB Throughput-Latency Tradeoff" \
								 --plot_throughput_vs_latency

# a100 and v100 Llama unquantized
python gpu_batch_exp_plotting.py --bmark_output_paths "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/7B/fp16/1-gpu-1-batch/bmark_numreqsample0_iter100_max1000_a10040gb.out" \
													  "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/7B/fp16/1-gpu-4-batch/bmark_numreqsample0_iter100_max1000_a10040gb.out" \
													  "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/7B/fp16/1-gpu-8-batch/bmark_numreqsample0_iter100_max1000_a10040gb.out" \
													  "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/7B/fp16/1-gpu-12-batch/bmark_numreqsample0_iter100_max1000_a10040gb.out" \
													  "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/7B/fp16/1-gpu-16-batch/bmark_numreqsample0_iter100_max1000_a10040gb.out" \
													  "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/7B/fp16/1-gpu-20-batch/bmark_numreqsample0_iter100_max1000_a10040gb.out" \
													  "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/7B/fp16/1-gpu-22-batch/bmark_numreqsample0_iter100_max1000_a10040gb.out" \
													  "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/13B/fp16/1-gpu-1-batch/bmark_numreqsample0_iter100_max1000_a10040gb.out" \
													  "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/13B/fp16/1-gpu-2-batch/bmark_numreqsample0_iter100_max1000_a10040gb.out" \
													  "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/13B/fp16/1-gpu-4-batch/bmark_numreqsample0_iter100_max1000_a10040gb.out" \
													  "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/13B/fp16/1-gpu-6-batch/bmark_numreqsample0_iter100_max1000_a10040gb.out" \
													  "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/13B/fp16/1-gpu-8-batch/bmark_numreqsample0_iter100_max1000_a10040gb.out" \
													  "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/7B/fp16/1-gpu-1-batch/bmark_numreqsample0_iter100_max1000_v10032gb.out" \
													  "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/7B/fp16/1-gpu-4-batch/bmark_numreqsample0_iter100_max1000_v10032gb.out" \
													  "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/7B/fp16/1-gpu-8-batch/bmark_numreqsample0_iter100_max1000_v10032gb.out" \
													  "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/7B/fp16/1-gpu-12-batch/bmark_numreqsample0_iter100_max1000_v10032gb.out" \
													  "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/7B/fp16/1-gpu-14-batch/bmark_numreqsample0_iter100_max1000_v10032gb.out" \
													  "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/13B/fp16/1-gpu-1-batch/bmark_numreqsample0_iter100_max1000_v10032gb.out" \
													  "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/13B/fp16/1-gpu-2-batch/bmark_numreqsample0_iter100_max1000_v10032gb.out" \
													  "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/13B/fp16/1-gpu-4-batch/bmark_numreqsample0_iter100_max1000_v10032gb.out" \
								 --nvsmi_output_paths "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/7B/fp16/1-gpu-1-batch/nvsmi_numreqsample0_iter100_max1000_a10040gb.out" \
													  "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/7B/fp16/1-gpu-4-batch/nvsmi_numreqsample0_iter100_max1000_a10040gb.out" \
													  "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/7B/fp16/1-gpu-8-batch/nvsmi_numreqsample0_iter100_max1000_a10040gb.out" \
													  "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/7B/fp16/1-gpu-12-batch/nvsmi_numreqsample0_iter100_max1000_a10040gb.out" \
													  "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/7B/fp16/1-gpu-16-batch/nvsmi_numreqsample0_iter100_max1000_a10040gb.out" \
													  "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/7B/fp16/1-gpu-20-batch/nvsmi_numreqsample0_iter100_max1000_a10040gb.out" \
													  "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/7B/fp16/1-gpu-22-batch/nvsmi_numreqsample0_iter100_max1000_a10040gb.out" \
													  "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/13B/fp16/1-gpu-1-batch/nvsmi_numreqsample0_iter100_max1000_a10040gb.out" \
													  "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/13B/fp16/1-gpu-2-batch/nvsmi_numreqsample0_iter100_max1000_a10040gb.out" \
													  "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/13B/fp16/1-gpu-4-batch/nvsmi_numreqsample0_iter100_max1000_a10040gb.out" \
													  "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/13B/fp16/1-gpu-6-batch/nvsmi_numreqsample0_iter100_max1000_a10040gb.out" \
													  "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/13B/fp16/1-gpu-8-batch/nvsmi_numreqsample0_iter100_max1000_a10040gb.out" \
													  "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/7B/fp16/1-gpu-1-batch/nvsmi_numreqsample0_iter100_max1000_v10032gb.out" \
													  "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/7B/fp16/1-gpu-4-batch/nvsmi_numreqsample0_iter100_max1000_v10032gb.out" \
													  "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/7B/fp16/1-gpu-8-batch/nvsmi_numreqsample0_iter100_max1000_v10032gb.out" \
													  "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/7B/fp16/1-gpu-12-batch/nvsmi_numreqsample0_iter100_max1000_v10032gb.out" \
													  "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/7B/fp16/1-gpu-14-batch/nvsmi_numreqsample0_iter100_max1000_v10032gb.out" \
													  "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/13B/fp16/1-gpu-1-batch/nvsmi_numreqsample0_iter100_max1000_v10032gb.out" \
													  "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/13B/fp16/1-gpu-2-batch/nvsmi_numreqsample0_iter100_max1000_v10032gb.out" \
													  "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/13B/fp16/1-gpu-4-batch/nvsmi_numreqsample0_iter100_max1000_v10032gb.out" \
								 --bmark_params       "7B 1 1000 a10040gb nowq" \
								                      "7B 4 1000 a10040gb nowq" \
													  "7B 8 1000 a10040gb nowq" \
													  "7B 12 1000 a10040gb nowq" \
													  "7B 16 1000 a10040gb nowq" \
													  "7B 20 1000 a10040gb nowq" \
													  "7B 22 1000 a10040gb nowq" \
													  "13B 1 1000 a10040gb nowq" \
													  "13B 2 1000 a10040gb nowq" \
													  "13B 4 1000 a10040gb nowq" \
													  "13B 6 1000 a10040gb nowq" \
													  "13B 8 1000 a10040gb nowq" \
													  "7B 1 1000 v10032gb nowq" \
													  "7B 4 1000 v10032gb nowq" \
													  "7B 8 1000 v10032gb nowq" \
													  "7B 12 1000 v10032gb nowq" \
													  "7B 14 1000 v10032gb nowq" \
													  "13B 1 1000 v10032gb nowq" \
													  "13B 2 1000 v10032gb nowq" \
													  "13B 4 1000 v10032gb nowq" \
								 --bmark_param_groups "7B X 1000 a10040gb nowq" \
								 					  "13B X 1000 a10040gb nowq" \
													  "7B X 1000 v10032gb nowq" \
													  "13B X 1000 v10032gb nowq" \
								 --plot_filename      "throughput_vs_latency_llama2_a10040gb_v10032gb.png" \
								 --plot_name          "Llama2 Throughput-Latency Tradeoff" \
								 --plot_throughput_vs_latency

# v100 gpt2 unquantized
python gpu_batch_exp_plotting.py --bmark_output_paths "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/gpt/137M/fp16/1-gpu-1-batch/bmark_numreqsample0_iter100_max500_v10032gb.out" \
													  "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/gpt/137M/fp16/1-gpu-32-batch/bmark_numreqsample0_iter100_max500_v10032gb.out" \
													  "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/gpt/137M/fp16/1-gpu-64-batch/bmark_numreqsample0_iter100_max500_v10032gb.out" \
													  "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/gpt/137M/fp16/1-gpu-96-batch/bmark_numreqsample0_iter100_max500_v10032gb.out" \
													  "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/gpt/137M/fp16/1-gpu-128-batch/bmark_numreqsample0_iter100_max500_v10032gb.out" \
													  "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/gpt/137M/fp16/1-gpu-256-batch/bmark_numreqsample0_iter100_max500_v10032gb.out" \
													  "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/gpt/137M/fp16/1-gpu-384-batch/bmark_numreqsample0_iter100_max500_v10032gb.out" \
													  "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/gpt/137M/fp16/1-gpu-512-batch/bmark_numreqsample0_iter100_max500_v10032gb.out" \
													  "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/gpt/137M/fp16/1-gpu-544-batch/bmark_numreqsample0_iter100_max500_v10032gb.out" \
													  "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/gpt/812M/fp16/1-gpu-1-batch/bmark_numreqsample0_iter100_max500_v10032gb.out" \
													  "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/gpt/812M/fp16/1-gpu-32-batch/bmark_numreqsample0_iter100_max500_v10032gb.out" \
													  "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/gpt/812M/fp16/1-gpu-64-batch/bmark_numreqsample0_iter100_max500_v10032gb.out" \
													  "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/gpt/812M/fp16/1-gpu-96-batch/bmark_numreqsample0_iter100_max500_v10032gb.out" \
													  "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/gpt/812M/fp16/1-gpu-128-batch/bmark_numreqsample0_iter100_max500_v10032gb.out" \
													  "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/gpt/2B/fp16/1-gpu-1-batch/bmark_numreqsample0_iter100_max500_v10032gb.out" \
													  "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/gpt/2B/fp16/1-gpu-16-batch/bmark_numreqsample0_iter100_max500_v10032gb.out" \
													  "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/gpt/2B/fp16/1-gpu-32-batch/bmark_numreqsample0_iter100_max500_v10032gb.out" \
													  "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/gpt/2B/fp16/1-gpu-48-batch/bmark_numreqsample0_iter100_max500_v10032gb.out" \
													  "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/gpt/2B/fp16/1-gpu-64-batch/bmark_numreqsample0_iter100_max500_v10032gb.out" \
													  "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/gpt/2B/fp16/1-gpu-72-batch/bmark_numreqsample0_iter100_max500_v10032gb.out" \
								 --nvsmi_output_paths "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/gpt/137M/fp16/1-gpu-1-batch/nvsmi_numreqsample0_iter100_max500_v10032gb.out" \
													  "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/gpt/137M/fp16/1-gpu-32-batch/nvsmi_numreqsample0_iter100_max500_v10032gb.out" \
													  "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/gpt/137M/fp16/1-gpu-64-batch/nvsmi_numreqsample0_iter100_max500_v10032gb.out" \
													  "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/gpt/137M/fp16/1-gpu-96-batch/nvsmi_numreqsample0_iter100_max500_v10032gb.out" \
													  "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/gpt/137M/fp16/1-gpu-128-batch/nvsmi_numreqsample0_iter100_max500_v10032gb.out" \
													  "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/gpt/137M/fp16/1-gpu-256-batch/nvsmi_numreqsample0_iter100_max500_v10032gb.out" \
													  "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/gpt/137M/fp16/1-gpu-384-batch/nvsmi_numreqsample0_iter100_max500_v10032gb.out" \
													  "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/gpt/137M/fp16/1-gpu-512-batch/nvsmi_numreqsample0_iter100_max500_v10032gb.out" \
													  "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/gpt/137M/fp16/1-gpu-544-batch/nvsmi_numreqsample0_iter100_max500_v10032gb.out" \
													  "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/gpt/812M/fp16/1-gpu-1-batch/nvsmi_numreqsample0_iter100_max500_v10032gb.out" \
													  "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/gpt/812M/fp16/1-gpu-32-batch/nvsmi_numreqsample0_iter100_max500_v10032gb.out" \
													  "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/gpt/812M/fp16/1-gpu-64-batch/nvsmi_numreqsample0_iter100_max500_v10032gb.out" \
													  "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/gpt/812M/fp16/1-gpu-96-batch/nvsmi_numreqsample0_iter100_max500_v10032gb.out" \
													  "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/gpt/812M/fp16/1-gpu-128-batch/nvsmi_numreqsample0_iter100_max500_v10032gb.out" \
													  "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/gpt/2B/fp16/1-gpu-1-batch/nvsmi_numreqsample0_iter100_max500_v10032gb.out" \
													  "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/gpt/2B/fp16/1-gpu-16-batch/nvsmi_numreqsample0_iter100_max500_v10032gb.out" \
													  "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/gpt/2B/fp16/1-gpu-32-batch/nvsmi_numreqsample0_iter100_max500_v10032gb.out" \
													  "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/gpt/2B/fp16/1-gpu-48-batch/nvsmi_numreqsample0_iter100_max500_v10032gb.out" \
													  "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/gpt/2B/fp16/1-gpu-64-batch/nvsmi_numreqsample0_iter100_max500_v10032gb.out" \
													  "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/gpt/2B/fp16/1-gpu-72-batch/nvsmi_numreqsample0_iter100_max500_v10032gb.out" \
								 --bmark_params       "137M 1 500 v10032gb nowq" \
								                      "137M 32 500 v10032gb nowq" \
								                      "137M 64 500 v10032gb nowq" \
								                      "137M 96 500 v10032gb nowq" \
								                      "137M 128 500 v10032gb nowq" \
								                      "137M 256 500 v10032gb nowq" \
								                      "137M 384 500 v10032gb nowq" \
								                      "137M 512 500 v10032gb nowq" \
								                      "137M 544 500 v10032gb nowq" \
													  "812M 1 500 v10032gb nowq" \
													  "812M 32 500 v10032gb nowq" \
													  "812M 64 500 v10032gb nowq" \
													  "812M 96 500 v10032gb nowq" \
													  "812M 128 500 v10032gb nowq" \
													  "2B 1 500 v10032gb nowq" \
													  "2B 16 500 v10032gb nowq" \
													  "2B 32 500 v10032gb nowq" \
													  "2B 48 500 v10032gb nowq" \
													  "2B 64 500 v10032gb nowq" \
													  "2B 72 500 v10032gb nowq" \
								 --bmark_param_groups "137M X 500 v10032gb nowq" \
								 					  "812M X 500 v10032gb nowq" \
													  "2B X 500 v10032gb nowq" \
								 --plot_filename      "throughput_vs_latency_gpt2_v10032gb.png" \
								 --plot_name          "GPT2 Throughput-Latency Tradeoff" \
								 --plot_throughput_vs_latency

# Llama2 unquantized vs. quantized for v10032gb
python gpu_batch_exp_plotting.py --bmark_output_paths "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/7B/fp16/1-gpu-1-batch/bmark_numreqsample0_iter100_max1000_v10032gb.out" \
													  "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/7B/fp16/1-gpu-4-batch/bmark_numreqsample0_iter100_max1000_v10032gb.out" \
													  "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/7B/fp16/1-gpu-8-batch/bmark_numreqsample0_iter100_max1000_v10032gb.out" \
													  "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/7B/fp16/1-gpu-12-batch/bmark_numreqsample0_iter100_max1000_v10032gb.out" \
													  "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/7B/fp16/1-gpu-14-batch/bmark_numreqsample0_iter100_max1000_v10032gb.out" \
													  "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/13B/fp16/1-gpu-1-batch/bmark_numreqsample0_iter100_max1000_v10032gb.out" \
													  "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/13B/fp16/1-gpu-2-batch/bmark_numreqsample0_iter100_max1000_v10032gb.out" \
													  "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/13B/fp16/1-gpu-4-batch/bmark_numreqsample0_iter100_max1000_v10032gb.out" \
													  "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/7B/fp16_wq4/1-gpu-1-batch/bmark_numreqsample0_iter100_max1000_v10032gb.out" \
													  "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/7B/fp16_wq4/1-gpu-4-batch/bmark_numreqsample0_iter100_max1000_v10032gb.out" \
													  "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/7B/fp16_wq4/1-gpu-8-batch/bmark_numreqsample0_iter100_max1000_v10032gb.out" \
													  "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/7B/fp16_wq4/1-gpu-12-batch/bmark_numreqsample0_iter100_max1000_v10032gb.out" \
													  "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/7B/fp16_wq4/1-gpu-16-batch/bmark_numreqsample0_iter100_max1000_v10032gb.out" \
													  "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/7B/fp16_wq4/1-gpu-20-batch/bmark_numreqsample0_iter100_max1000_v10032gb.out" \
													  "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/7B/fp16_wq4/1-gpu-22-batch/bmark_numreqsample0_iter100_max1000_v10032gb.out" \
													  "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/13B/fp16_wq4/1-gpu-1-batch/bmark_numreqsample0_iter100_max1000_v10032gb.out" \
													  "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/13B/fp16_wq4/1-gpu-2-batch/bmark_numreqsample0_iter100_max1000_v10032gb.out" \
													  "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/13B/fp16_wq4/1-gpu-4-batch/bmark_numreqsample0_iter100_max1000_v10032gb.out" \
													  "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/13B/fp16_wq4/1-gpu-6-batch/bmark_numreqsample0_iter100_max1000_v10032gb.out" \
													  "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/13B/fp16_wq4/1-gpu-8-batch/bmark_numreqsample0_iter100_max1000_v10032gb.out" \
													  "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/13B/fp16_wq4/1-gpu-10-batch/bmark_numreqsample0_iter100_max1000_v10032gb.out" \
													  "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/13B/fp16_wq4/1-gpu-12-batch/bmark_numreqsample0_iter100_max1000_v10032gb.out" \
								 --nvsmi_output_paths "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/7B/fp16/1-gpu-1-batch/nvsmi_numreqsample0_iter100_max1000_v10032gb.out" \
													  "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/7B/fp16/1-gpu-4-batch/nvsmi_numreqsample0_iter100_max1000_v10032gb.out" \
													  "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/7B/fp16/1-gpu-8-batch/nvsmi_numreqsample0_iter100_max1000_v10032gb.out" \
													  "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/7B/fp16/1-gpu-12-batch/nvsmi_numreqsample0_iter100_max1000_v10032gb.out" \
													  "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/7B/fp16/1-gpu-14-batch/nvsmi_numreqsample0_iter100_max1000_v10032gb.out" \
													  "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/13B/fp16/1-gpu-1-batch/nvsmi_numreqsample0_iter100_max1000_v10032gb.out" \
													  "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/13B/fp16/1-gpu-2-batch/nvsmi_numreqsample0_iter100_max1000_v10032gb.out" \
													  "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/13B/fp16/1-gpu-4-batch/nvsmi_numreqsample0_iter100_max1000_v10032gb.out" \
													  "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/7B/fp16_wq4/1-gpu-1-batch/nvsmi_numreqsample0_iter100_max1000_v10032gb.out" \
													  "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/7B/fp16_wq4/1-gpu-4-batch/nvsmi_numreqsample0_iter100_max1000_v10032gb.out" \
													  "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/7B/fp16_wq4/1-gpu-8-batch/nvsmi_numreqsample0_iter100_max1000_v10032gb.out" \
													  "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/7B/fp16_wq4/1-gpu-12-batch/nvsmi_numreqsample0_iter100_max1000_v10032gb.out" \
													  "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/7B/fp16_wq4/1-gpu-16-batch/nvsmi_numreqsample0_iter100_max1000_v10032gb.out" \
													  "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/7B/fp16_wq4/1-gpu-20-batch/nvsmi_numreqsample0_iter100_max1000_v10032gb.out" \
													  "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/7B/fp16_wq4/1-gpu-22-batch/nvsmi_numreqsample0_iter100_max1000_v10032gb.out" \
													  "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/13B/fp16_wq4/1-gpu-1-batch/nvsmi_numreqsample0_iter100_max1000_v10032gb.out" \
													  "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/13B/fp16_wq4/1-gpu-2-batch/nvsmi_numreqsample0_iter100_max1000_v10032gb.out" \
													  "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/13B/fp16_wq4/1-gpu-4-batch/nvsmi_numreqsample0_iter100_max1000_v10032gb.out" \
													  "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/13B/fp16_wq4/1-gpu-6-batch/nvsmi_numreqsample0_iter100_max1000_v10032gb.out" \
													  "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/13B/fp16_wq4/1-gpu-8-batch/nvsmi_numreqsample0_iter100_max1000_v10032gb.out" \
													  "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/13B/fp16_wq4/1-gpu-10-batch/nvsmi_numreqsample0_iter100_max1000_v10032gb.out" \
													  "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/13B/fp16_wq4/1-gpu-12-batch/nvsmi_numreqsample0_iter100_max1000_v10032gb.out" \
								 --bmark_params		  "7B 1 1000 v10032gb nowq" \
													  "7B 4 1000 v10032gb nowq" \
													  "7B 8 1000 v10032gb nowq" \
													  "7B 12 1000 v10032gb nowq" \
													  "7B 14 1000 v10032gb nowq" \
													  "13B 1 1000 v10032gb nowq" \
													  "13B 2 1000 v10032gb nowq" \
													  "13B 4 1000 v10032gb nowq" \
													  "7B 1 1000 v10032gb wq4" \
													  "7B 4 1000 v10032gb wq4" \
													  "7B 8 1000 v10032gb wq4" \
													  "7B 12 1000 v10032gb wq4" \
													  "7B 16 1000 v10032gb wq4" \
													  "7B 20 1000 v10032gb wq4" \
													  "7B 22 1000 v10032gb wq4" \
													  "13B 1 1000 v10032gb wq4" \
													  "13B 2 1000 v10032gb wq4" \
													  "13B 4 1000 v10032gb wq4" \
													  "13B 6 1000 v10032gb wq4" \
													  "13B 8 1000 v10032gb wq4" \
													  "13B 10 1000 v10032gb wq4" \
													  "13B 12 1000 v10032gb wq4" \
								 --bmark_param_groups "7B X 1000 v10032gb nowq" \
								  					  "13B X 1000 v10032gb nowq" \
													  "7B X 1000 v10032gb wq4" \
													  "13B X 1000 v10032gb wq4" \
								 --plot_filename      "throughput_vs_latency_llama2_v10032gb_quantization.png" \
								 --plot_name          "Llama2 Throughput-Latency Tradeoff w/ Weight Quantization" \
								 --plot_throughput_vs_latency


python fix_nvsmi_output_timing.py --bmark_output_path  "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/7B/fp16/1-gpu-4-batch/bmark_numreqsample0_iter100_max1000_a10040gb.out" \
								  --nvsmi_output_path  "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/7B/fp16/1-gpu-4-batch/nvsmi_numreqsample0_iter100_max1000_a10040gb.out" \
								  --gpu_idx            0 \
								  --nvsmi_start_line   38 \
								  --nvsmi_end_line     319

python fix_nvsmi_output_timing.py --bmark_output_path  "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/7B/fp16/1-gpu-8-batch/bmark_numreqsample0_iter100_max1000_a10040gb.out" \
								  --nvsmi_output_path  "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/7B/fp16/1-gpu-8-batch/nvsmi_numreqsample0_iter100_max1000_a10040gb.out" \
								  --gpu_idx            0 \
								  --nvsmi_start_line   37 \
								  --nvsmi_end_line     422

python fix_nvsmi_output_timing.py --bmark_output_path  "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/7B/fp16/1-gpu-12-batch/bmark_numreqsample0_iter100_max1000_a10040gb.out" \
								  --nvsmi_output_path  "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/7B/fp16/1-gpu-12-batch/nvsmi_numreqsample0_iter100_max1000_a10040gb.out" \
								  --gpu_idx            0 \
								  --nvsmi_start_line   37 \
								  --nvsmi_end_line     505

python fix_nvsmi_output_timing.py --bmark_output_path  "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/7B/fp16/1-gpu-16-batch/bmark_numreqsample0_iter100_max1000_a10040gb.out" \
								  --nvsmi_output_path  "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/7B/fp16/1-gpu-16-batch/nvsmi_numreqsample0_iter100_max1000_a10040gb.out" \
								  --gpu_idx            0 \
								  --nvsmi_start_line   37 \
								  --nvsmi_end_line     620

python fix_nvsmi_output_timing.py --bmark_output_path  "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/7B/fp16/1-gpu-20-batch/bmark_numreqsample0_iter100_max1000_a10040gb.out" \
								  --nvsmi_output_path  "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/7B/fp16/1-gpu-20-batch/nvsmi_numreqsample0_iter100_max1000_a10040gb.out" \
								  --gpu_idx            0 \
								  --nvsmi_start_line   36 \
								  --nvsmi_end_line     684

python fix_nvsmi_output_timing.py --bmark_output_path  "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/7B/fp16/1-gpu-22-batch/bmark_numreqsample0_iter100_max1000_a10040gb.out" \
								  --nvsmi_output_path  "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/7B/fp16/1-gpu-22-batch/nvsmi_numreqsample0_iter100_max1000_a10040gb.out" \
								  --gpu_idx            0 \
								  --nvsmi_start_line   36 \
								  --nvsmi_end_line     720

python fix_nvsmi_output_timing.py --bmark_output_path  "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/7B/fp16_wq4/1-gpu-1-batch/bmark_numreqsample0_iter100_max1000_a10040gb.out" \
								  --nvsmi_output_path  "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/7B/fp16_wq4/1-gpu-1-batch/nvsmi_numreqsample0_iter100_max1000_a10040gb.out" \
								  --gpu_idx            0 \
								  --nvsmi_start_line   36 \
								  --nvsmi_end_line     88

python fix_nvsmi_output_timing.py --bmark_output_path  "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/7B/fp16_wq4/1-gpu-1-batch/bmark_numreqsample0_iter100_max1000_v10032gb.out" \
								  --nvsmi_output_path  "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/7B/fp16_wq4/1-gpu-1-batch/nvsmi_numreqsample0_iter100_max1000_v10032gb.out" \
								  --gpu_idx            0 \
								  --nvsmi_start_line   74 \
								  --nvsmi_end_line     186

python fix_nvsmi_output_timing.py --bmark_output_path  "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/7B/fp16_wq4/1-gpu-4-batch/bmark_numreqsample0_iter100_max1000_a10040gb.out" \
								  --nvsmi_output_path  "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/7B/fp16_wq4/1-gpu-4-batch/nvsmi_numreqsample0_iter100_max1000_a10040gb.out" \
								  --gpu_idx            0 \
								  --nvsmi_start_line   37 \
								  --nvsmi_end_line     208

python fix_nvsmi_output_timing.py --bmark_output_path  "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/7B/fp16_wq4/1-gpu-4-batch/bmark_numreqsample0_iter100_max1000_v10032gb.out" \
								  --nvsmi_output_path  "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/7B/fp16_wq4/1-gpu-4-batch/nvsmi_numreqsample0_iter100_max1000_v10032gb.out" \
								  --gpu_idx            0 \
								  --nvsmi_start_line   72 \
								  --nvsmi_end_line     492

python fix_nvsmi_output_timing.py --bmark_output_path  "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/7B/fp16_wq4/1-gpu-8-batch/bmark_numreqsample0_iter100_max1000_a10040gb.out" \
								  --nvsmi_output_path  "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/7B/fp16_wq4/1-gpu-8-batch/nvsmi_numreqsample0_iter100_max1000_a10040gb.out" \
								  --gpu_idx            0 \
								  --nvsmi_start_line   40 \
								  --nvsmi_end_line     282

python fix_nvsmi_output_timing.py --bmark_output_path  "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/7B/fp16_wq4/1-gpu-8-batch/bmark_numreqsample0_iter100_max1000_v10032gb.out" \
								  --nvsmi_output_path  "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/7B/fp16_wq4/1-gpu-8-batch/nvsmi_numreqsample0_iter100_max1000_v10032gb.out" \
								  --gpu_idx            0 \
								  --nvsmi_start_line   74 \
								  --nvsmi_end_line     1513

python fix_nvsmi_output_timing.py --bmark_output_path  "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/7B/fp16_wq4/1-gpu-12-batch/bmark_numreqsample0_iter100_max1000_a10040gb.out" \
								  --nvsmi_output_path  "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/7B/fp16_wq4/1-gpu-12-batch/nvsmi_numreqsample0_iter100_max1000_a10040gb.out" \
								  --gpu_idx            0 \
								  --nvsmi_start_line   35 \
								  --nvsmi_end_line     364

python fix_nvsmi_output_timing.py --bmark_output_path  "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/7B/fp16_wq4/1-gpu-12-batch/bmark_numreqsample0_iter100_max1000_v10032gb.out" \
								  --nvsmi_output_path  "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/7B/fp16_wq4/1-gpu-12-batch/nvsmi_numreqsample0_iter100_max1000_v10032gb.out" \
								  --gpu_idx            0 \
								  --nvsmi_start_line   72 \
								  --nvsmi_end_line     1904

python fix_nvsmi_output_timing.py --bmark_output_path  "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/7B/fp16_wq4/1-gpu-16-batch/bmark_numreqsample0_iter100_max1000_v10032gb.out" \
								  --nvsmi_output_path  "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/7B/fp16_wq4/1-gpu-16-batch/nvsmi_numreqsample0_iter100_max1000_v10032gb.out" \
								  --gpu_idx            0 \
								  --nvsmi_start_line   72 \
								  --nvsmi_end_line     2256

python fix_nvsmi_output_timing.py --bmark_output_path  "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/7B/fp16_wq4/1-gpu-20-batch/bmark_numreqsample0_iter100_max1000_v10032gb.out" \
								  --nvsmi_output_path  "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/7B/fp16_wq4/1-gpu-20-batch/nvsmi_numreqsample0_iter100_max1000_v10032gb.out" \
								  --gpu_idx            0 \
								  --nvsmi_start_line   73 \
								  --nvsmi_end_line     2553

python fix_nvsmi_output_timing.py --bmark_output_path  "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/7B/fp16_wq4/1-gpu-22-batch/bmark_numreqsample0_iter100_max1000_v10032gb.out" \
								  --nvsmi_output_path  "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/7B/fp16_wq4/1-gpu-22-batch/nvsmi_numreqsample0_iter100_max1000_v10032gb.out" \
								  --gpu_idx            0 \
								  --nvsmi_start_line   72 \
								  --nvsmi_end_line     2772

python fix_nvsmi_output_timing.py --bmark_output_path "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/13B/fp16/1-gpu-1-batch/bmark_numreqsample0_iter100_max1000_a10040gb.out" \
								  --nvsmi_output_path "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/13B/fp16/1-gpu-1-batch/nvsmi_numreqsample0_iter100_max1000_a10040gb.out" \
								  --gpu_idx			  0 \
								  --nvsmi_start_line  38 \
								  --nvsmi_end_line    248

python fix_nvsmi_output_timing.py --bmark_output_path "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/13B/fp16/1-gpu-2-batch/bmark_numreqsample0_iter100_max1000_a10040gb.out" \
								  --nvsmi_output_path "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/13B/fp16/1-gpu-2-batch/nvsmi_numreqsample0_iter100_max1000_a10040gb.out" \
								  --gpu_idx			  0 \
								  --nvsmi_start_line  38 \
								  --nvsmi_end_line    348

python fix_nvsmi_output_timing.py --bmark_output_path "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/13B/fp16/1-gpu-2-batch/bmark_numreqsample0_iter100_max1000_a10040gb.out" \
								  --nvsmi_output_path "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/13B/fp16/1-gpu-2-batch/nvsmi_numreqsample0_iter100_max1000_a10040gb.out" \
								  --gpu_idx			  0 \
								  --nvsmi_start_line  38 \
								  --nvsmi_end_line    348

python fix_nvsmi_output_timing.py --bmark_output_path "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/13B/fp16/1-gpu-4-batch/bmark_numreqsample0_iter100_max1000_a10040gb.out" \
								  --nvsmi_output_path "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/13B/fp16/1-gpu-4-batch/nvsmi_numreqsample0_iter100_max1000_a10040gb.out" \
								  --gpu_idx			  0 \
								  --nvsmi_start_line  37 \
								  --nvsmi_end_line    536

python fix_nvsmi_output_timing.py --bmark_output_path "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/13B/fp16/1-gpu-6-batch/bmark_numreqsample0_iter100_max1000_a10040gb.out" \
								  --nvsmi_output_path "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/13B/fp16/1-gpu-6-batch/nvsmi_numreqsample0_iter100_max1000_a10040gb.out" \
								  --gpu_idx			  0 \
								  --nvsmi_start_line  38 \
								  --nvsmi_end_line    652

python fix_nvsmi_output_timing.py --bmark_output_path "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/13B/fp16/1-gpu-8-batch/bmark_numreqsample0_iter100_max1000_a10040gb.out" \
								  --nvsmi_output_path "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/13B/fp16/1-gpu-8-batch/nvsmi_numreqsample0_iter100_max1000_a10040gb.out" \
								  --gpu_idx			  0 \
								  --nvsmi_start_line  37 \
								  --nvsmi_end_line    729

python fix_nvsmi_output_timing.py --bmark_output_path "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/13B/fp16_wq4/1-gpu-1-batch/bmark_numreqsample0_iter100_max1000_v10032gb.out" \
								  --nvsmi_output_path "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/13B/fp16_wq4/1-gpu-1-batch/nvsmi_numreqsample0_iter100_max1000_v10032gb.out" \
								  --gpu_idx			  0 \
								  --nvsmi_start_line  74 \
								  --nvsmi_end_line    253

python fix_nvsmi_output_timing.py --bmark_output_path "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/13B/fp16_wq4/1-gpu-2-batch/bmark_numreqsample0_iter100_max1000_v10032gb.out" \
								  --nvsmi_output_path "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/13B/fp16_wq4/1-gpu-2-batch/nvsmi_numreqsample0_iter100_max1000_v10032gb.out" \
								  --gpu_idx			  0 \
								  --nvsmi_start_line  71 \
								  --nvsmi_end_line    429

python fix_nvsmi_output_timing.py --bmark_output_path "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/13B/fp16_wq4/1-gpu-4-batch/bmark_numreqsample0_iter100_max1000_v10032gb.out" \
								  --nvsmi_output_path "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/13B/fp16_wq4/1-gpu-4-batch/nvsmi_numreqsample0_iter100_max1000_v10032gb.out" \
								  --gpu_idx			  0 \
								  --nvsmi_start_line  73 \
								  --nvsmi_end_line    780

python fix_nvsmi_output_timing.py --bmark_output_path "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/13B/fp16_wq4/1-gpu-6-batch/bmark_numreqsample0_iter100_max1000_v10032gb.out" \
								  --nvsmi_output_path "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/13B/fp16_wq4/1-gpu-6-batch/nvsmi_numreqsample0_iter100_max1000_v10032gb.out" \
								  --gpu_idx			  0 \
								  --nvsmi_start_line  75 \
								  --nvsmi_end_line    2271

python fix_nvsmi_output_timing.py --bmark_output_path "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/13B/fp16_wq4/1-gpu-8-batch/bmark_numreqsample0_iter100_max1000_v10032gb.out" \
								  --nvsmi_output_path "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/13B/fp16_wq4/1-gpu-8-batch/nvsmi_numreqsample0_iter100_max1000_v10032gb.out" \
								  --gpu_idx			  0 \
								  --nvsmi_start_line  74 \
								  --nvsmi_end_line    2595

python fix_nvsmi_output_timing.py --bmark_output_path "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/13B/fp16_wq4/1-gpu-10-batch/bmark_numreqsample0_iter100_max1000_v10032gb.out" \
								  --nvsmi_output_path "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/13B/fp16_wq4/1-gpu-10-batch/nvsmi_numreqsample0_iter100_max1000_v10032gb.out" \
								  --gpu_idx			  0 \
								  --nvsmi_start_line  72 \
								  --nvsmi_end_line    2874

python fix_nvsmi_output_timing.py --bmark_output_path "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/13B/fp16_wq4/1-gpu-12-batch/bmark_numreqsample0_iter100_max1000_v10032gb.out" \
								  --nvsmi_output_path "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/13B/fp16_wq4/1-gpu-12-batch/nvsmi_numreqsample0_iter100_max1000_v10032gb.out" \
								  --gpu_idx			  0 \
								  --nvsmi_start_line  73 \
								  --nvsmi_end_line    3233

# power plots for throughput-optimal vs. latency-optimal llama2
python3 gpu_batch_exp_plotting.py --bmark_output_paths "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/7B/fp16/1-gpu-1-batch/bmark_numreqsample0_iter100_max1000_a10040gb.out" \
													   "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/7B/fp16/1-gpu-22-batch/bmark_numreqsample0_iter100_max1000_a10040gb.out" \
													   "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/13B/fp16/1-gpu-1-batch/bmark_numreqsample0_iter100_max1000_a10040gb.out" \
													   "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/13B/fp16/1-gpu-8-batch/bmark_numreqsample0_iter100_max1000_a10040gb.out" \
													   "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/7B/fp16/1-gpu-1-batch/bmark_numreqsample0_iter100_max1000_v10032gb.out" \
													   "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/7B/fp16/1-gpu-14-batch/bmark_numreqsample0_iter100_max1000_v10032gb.out" \
													   "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/13B/fp16/1-gpu-1-batch/bmark_numreqsample0_iter100_max1000_v10032gb.out" \
													   "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/13B/fp16/1-gpu-4-batch/bmark_numreqsample0_iter100_max1000_v10032gb.out" \
								  --nvsmi_output_paths "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/7B/fp16/1-gpu-1-batch/nvsmi_numreqsample0_iter100_max1000_a10040gb.out" \
													   "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/7B/fp16/1-gpu-22-batch/nvsmi_numreqsample0_iter100_max1000_a10040gb.out" \
													   "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/13B/fp16/1-gpu-1-batch/nvsmi_numreqsample0_iter100_max1000_a10040gb.out" \
													   "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/13B/fp16/1-gpu-8-batch/nvsmi_numreqsample0_iter100_max1000_a10040gb.out" \
													   "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/7B/fp16/1-gpu-1-batch/nvsmi_numreqsample0_iter100_max1000_v10032gb.out" \
													   "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/7B/fp16/1-gpu-14-batch/nvsmi_numreqsample0_iter100_max1000_v10032gb.out" \
													   "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/13B/fp16/1-gpu-1-batch/nvsmi_numreqsample0_iter100_max1000_v10032gb.out" \
													   "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/13B/fp16/1-gpu-4-batch/nvsmi_numreqsample0_iter100_max1000_v10032gb.out" \
								  --bmark_params       "7B 1 1000 a10040gb nowq" \
								  					   "7B 22 1000 a10040gb nowq" \
													   "13B 1 1000 a10040gb nowq" \
													   "13B 8 1000 a10040gb nowq" \
													   "7B 1 1000 v10032gb nowq" \
													   "7B 14 1000 v10032gb nowq" \
													   "13B 1 1000 v10032gb nowq" \
													   "13b 4 1000 v10032gb nowq" \
								  --gpu_idx			   0 \
								  --plot_filename      "throughput_opt_vs_latency_opt_power.png" \
								  --plot_name		   "Thoughput-optimized vs. Latency-optimized Power Profile" \
								  --plot_power_or_energy \
								  --project_24_hr

python fix_nvsmi_output_timing.py --bmark_output_path "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/gpt/137M/fp16/1-gpu-1-batch/bmark_numreqsample0_iter100_max500_v10032gb.out" \
								  --nvsmi_output_path "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/gpt/137M/fp16/1-gpu-1-batch/nvsmi_numreqsample0_iter100_max500_v10032gb.out" \
								  --gpu_idx           0 \
								  --nvsmi_start_line  27 \
								  --nvsmi_end_line    46

python fix_nvsmi_output_timing.py --bmark_output_path "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/gpt/137M/fp16/1-gpu-544-batch/bmark_numreqsample0_iter100_max500_v10032gb.out" \
								  --nvsmi_output_path "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/gpt/137M/fp16/1-gpu-544-batch/nvsmi_numreqsample0_iter100_max500_v10032gb.out" \
								  --gpu_idx           0 \
								  --nvsmi_start_line  38 \
								  --nvsmi_end_line    918

python fix_nvsmi_output_timing.py --bmark_output_path "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/gpt/812M/fp16/1-gpu-1-batch/bmark_numreqsample0_iter100_max500_v10032gb.out" \
								  --nvsmi_output_path "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/gpt/812M/fp16/1-gpu-1-batch/nvsmi_numreqsample0_iter100_max500_v10032gb.out" \
								  --gpu_idx           0 \
								  --nvsmi_start_line  25 \
								  --nvsmi_end_line    88

python fix_nvsmi_output_timing.py --bmark_output_path "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/gpt/812M/fp16/1-gpu-128-batch/bmark_numreqsample0_iter100_max500_v10032gb.out" \
								  --nvsmi_output_path "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/gpt/812M/fp16/1-gpu-128-batch/nvsmi_numreqsample0_iter100_max500_v10032gb.out" \
								  --gpu_idx           0 \
								  --nvsmi_start_line  29 \
								  --nvsmi_end_line    1159

python fix_nvsmi_output_timing.py --bmark_output_path "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/gpt/2B/fp16/1-gpu-1-batch/bmark_numreqsample0_iter100_max500_v10032gb.out" \
								  --nvsmi_output_path "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/gpt/2B/fp16/1-gpu-1-batch/nvsmi_numreqsample0_iter100_max500_v10032gb.out" \
								  --gpu_idx           0 \
								  --nvsmi_start_line  17 \
								  --nvsmi_end_line    138

python fix_nvsmi_output_timing.py --bmark_output_path "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/gpt/2B/fp16/1-gpu-72-batch/bmark_numreqsample0_iter100_max500_v10032gb.out" \
								  --nvsmi_output_path "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/gpt/2B/fp16/1-gpu-72-batch/nvsmi_numreqsample0_iter100_max500_v10032gb.out" \
								  --gpu_idx           0 \
								  --nvsmi_start_line  13 \
								  --nvsmi_end_line    1302

# power plots for throughput-optimal vs. latency-optimal gpt2
python3 gpu_batch_exp_plotting.py --bmark_output_paths "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/gpt/137M/fp16/1-gpu-544-batch/bmark_numreqsample0_iter100_max500_v10032gb.out" \
													   "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/gpt/812M/fp16/1-gpu-1-batch/bmark_numreqsample0_iter100_max500_v10032gb.out" \
													   "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/gpt/812M/fp16/1-gpu-128-batch/bmark_numreqsample0_iter100_max500_v10032gb.out" \
													   "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/gpt/2B/fp16/1-gpu-1-batch/bmark_numreqsample0_iter100_max500_v10032gb.out" \
													   "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/gpt/2B/fp16/1-gpu-72-batch/bmark_numreqsample0_iter100_max500_v10032gb.out" \
								  --nvsmi_output_paths "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/gpt/137M/fp16/1-gpu-544-batch/nvsmi_numreqsample0_iter100_max500_v10032gb.out" \
													   "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/gpt/812M/fp16/1-gpu-1-batch/nvsmi_numreqsample0_iter100_max500_v10032gb.out" \
													   "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/gpt/812M/fp16/1-gpu-128-batch/nvsmi_numreqsample0_iter100_max500_v10032gb.out" \
													   "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/gpt/2B/fp16/1-gpu-1-batch/nvsmi_numreqsample0_iter100_max500_v10032gb.out" \
													   "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/gpt/2B/fp16/1-gpu-72-batch/nvsmi_numreqsample0_iter100_max500_v10032gb.out" \
								  --bmark_params       "137M 544 500 v10032gb nowq" \
													   "812M 1 500 v10032gb nowq" \
													   "812M 128 500 v10032gb nowq" \
													   "2B 1 500 v10032gb nowq" \
													   "2B 72 500 v10032gb nowq" \
								  --gpu_idx			   0 \
								  --plot_filename      "gpt_throughput_opt_vs_latency_opt_power.png" \
								  --plot_name		   "" \
								  --plot_power_or_energy \
								  --project_24_hr

# power plots for throughput-optimal vs. latency-optimal gpt2
python3 gpu_batch_exp_plotting.py --bmark_output_paths "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/gpt/137M/fp16/1-gpu-1-batch/bmark_numreqsample0_iter100_max500_v10032gb.out" \
													   "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/gpt/137M/fp16/1-gpu-544-batch/bmark_numreqsample0_iter100_max500_v10032gb.out" \
													   "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/gpt/812M/fp16/1-gpu-1-batch/bmark_numreqsample0_iter100_max500_v10032gb.out" \
													   "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/gpt/812M/fp16/1-gpu-128-batch/bmark_numreqsample0_iter100_max500_v10032gb.out" \
													   "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/gpt/2B/fp16/1-gpu-1-batch/bmark_numreqsample0_iter100_max500_v10032gb.out" \
													   "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/gpt/2B/fp16/1-gpu-72-batch/bmark_numreqsample0_iter100_max500_v10032gb.out" \
								  --nvsmi_output_paths "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/gpt/137M/fp16/1-gpu-1-batch/nvsmi_numreqsample0_iter100_max500_v10032gb.out" \
								  					   "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/gpt/137M/fp16/1-gpu-544-batch/nvsmi_numreqsample0_iter100_max500_v10032gb.out" \
													   "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/gpt/812M/fp16/1-gpu-1-batch/nvsmi_numreqsample0_iter100_max500_v10032gb.out" \
													   "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/gpt/812M/fp16/1-gpu-128-batch/nvsmi_numreqsample0_iter100_max500_v10032gb.out" \
													   "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/gpt/2B/fp16/1-gpu-1-batch/nvsmi_numreqsample0_iter100_max500_v10032gb.out" \
													   "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/gpt/2B/fp16/1-gpu-72-batch/nvsmi_numreqsample0_iter100_max500_v10032gb.out" \
								  --bmark_params       "137M 1 500 v10032gb nowq" \
								  					   "137M 544 500 v10032gb nowq" \
													   "812M 1 500 v10032gb nowq" \
													   "812M 128 500 v10032gb nowq" \
													   "2B 1 500 v10032gb nowq" \
													   "2B 72 500 v10032gb nowq" \
								  --gpu_idx			   0 \
								  --plot_filename      "gpt_energy_per_token.png" \
								  --plot_name		   "GPT2 Joules Per Token" \
								  --plot_power_or_energy \
								  --plot_token_energy



# power plots for throughput-optimal vs. latency-optimal llama2
python3 gpu_batch_exp_plotting.py --bmark_output_paths "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/7B/fp16/1-gpu-1-batch/bmark_numreqsample0_iter100_max1000_a10040gb.out" \
													   "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/7B/fp16/1-gpu-22-batch/bmark_numreqsample0_iter100_max1000_a10040gb.out" \
													   "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/13B/fp16/1-gpu-1-batch/bmark_numreqsample0_iter100_max1000_a10040gb.out" \
													   "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/13B/fp16/1-gpu-8-batch/bmark_numreqsample0_iter100_max1000_a10040gb.out" \
													   "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/7B/fp16/1-gpu-1-batch/bmark_numreqsample0_iter100_max1000_v10032gb.out" \
													   "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/7B/fp16/1-gpu-14-batch/bmark_numreqsample0_iter100_max1000_v10032gb.out" \
													   "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/13B/fp16/1-gpu-1-batch/bmark_numreqsample0_iter100_max1000_v10032gb.out" \
													   "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/13B/fp16/1-gpu-4-batch/bmark_numreqsample0_iter100_max1000_v10032gb.out" \
								  --nvsmi_output_paths "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/7B/fp16/1-gpu-1-batch/nvsmi_numreqsample0_iter100_max1000_a10040gb.out" \
													   "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/7B/fp16/1-gpu-22-batch/nvsmi_numreqsample0_iter100_max1000_a10040gb.out" \
													   "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/13B/fp16/1-gpu-1-batch/nvsmi_numreqsample0_iter100_max1000_a10040gb.out" \
													   "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/13B/fp16/1-gpu-8-batch/nvsmi_numreqsample0_iter100_max1000_a10040gb.out" \
													   "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/7B/fp16/1-gpu-1-batch/nvsmi_numreqsample0_iter100_max1000_v10032gb.out" \
													   "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/7B/fp16/1-gpu-14-batch/nvsmi_numreqsample0_iter100_max1000_v10032gb.out" \
													   "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/13B/fp16/1-gpu-1-batch/nvsmi_numreqsample0_iter100_max1000_v10032gb.out" \
													   "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/13B/fp16/1-gpu-4-batch/nvsmi_numreqsample0_iter100_max1000_v10032gb.out" \
								  --bmark_params       "7B 1 1000 a10040gb nowq" \
								  					   "7B 22 1000 a10040gb nowq" \
													   "13B 1 1000 a10040gb nowq" \
													   "13B 8 1000 a10040gb nowq" \
													   "7B 1 1000 v10032gb nowq" \
													   "7B 14 1000 v10032gb nowq" \
													   "13B 1 1000 v10032gb nowq" \
													   "13b 4 1000 v10032gb nowq" \
								  --gpu_idx			   0 \
								  --plot_filename      "llama_energy_per_token.png" \
								  --plot_name		   "Llama2 Joules Per Token" \
								  --plot_power_or_energy \
								  --plot_token_energy
