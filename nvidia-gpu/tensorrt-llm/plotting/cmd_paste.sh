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

# just a100
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

# a100 and v100 Llama
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
													  "7B 1 1000 v10032gb" \
													  "7B 4 1000 v10032gb" \
													  "7B 8 1000 v10032gb" \
													  "7B 12 1000 v10032gb" \
													  "7B 14 1000 v10032gb" \
													  "13B 1 1000 v10032gb" \
													  "13B 2 1000 v10032gb" \
													  "13B 4 1000 v10032gb" \
								 --bmark_param_groups "7B X 1000 a10040gb" \
								 					  "13B X 1000 a10040gb" \
													  "7B X 1000 v10032gb" \
													  "13B X 1000 v10032gb" \
								 --plot_filename      "throughput_vs_latency_llama2_a10040gb_v10032gb.png" \
								 --plot_name          "Llama2 Throughput-Latency Tradeoff" \
								 --plot_throughput_vs_latency
