python gpu_batch_exp_plotting.py --bmark_output_paths "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/7B/fp16/1-gpu-1-batch/bmark_numreqsample10000_iter100.out /dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/7B/fp16/1-gpu-2-batch/bmark_numreqsample10000_iter100.out /dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/7B/fp16/1-gpu-4-batch/bmark_numreqsample10000_iter100.out /dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/7B/fp16/1-gpu-8-batch/bmark_numreqsample10000_iter100.out /dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/7B/fp16/1-gpu-12-batch/bmark_numreqsample10000_iter100.out" --nvsmi_output_paths "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/7B/fp16/1-gpu-1-batch/nvsmi_numreqsample10000_iter100.out /dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/7B/fp16/1-gpu-2-batch/nvsmi_numreqsample10000_iter100.out /dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/7B/fp16/1-gpu-4-batch/nvsmi_numreqsample10000_iter100.out /dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/7B/fp16/1-gpu-8-batch/nvsmi_numreqsample10000_iter100.out /dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/7B/fp16/1-gpu-12-batch/nvsmi_numreqsample10000_iter100.out"

python gpu_batch_exp_plotting.py --bmark_output_paths "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/13B/fp16/1-gpu-1-batch/bmark_numreqsample0_iter100_max1000.out" \
								 					  "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/13B/fp16/1-gpu-1-batch/bmark_numreqsample0_iter100_max500.out" \
													  "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/13B/fp16/1-gpu-2-batch/bmark_numreqsample0_iter100_max1000.out" \
								 					  "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/13B/fp16/1-gpu-2-batch/bmark_numreqsample0_iter100_max500.out" \
													  "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/13B/fp16/1-gpu-4-batch/bmark_numreqsample0_iter100_max1000.out" \
								 					  "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/13B/fp16/1-gpu-4-batch/bmark_numreqsample0_iter100_max500.out" \
								 					  "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/13B/fp16/1-gpu-6-batch/bmark_numreqsample0_iter100_max500.out" \
								 --nvsmi_output_paths "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/13B/fp16/1-gpu-1-batch/nvsmi_numreqsample0_iter100_max1000.out" \
								 					  "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/13B/fp16/1-gpu-1-batch/nvsmi_numreqsample0_iter100_max500.out" \
													  "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/13B/fp16/1-gpu-2-batch/nvsmi_numreqsample0_iter100_max1000.out" \
								 					  "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/13B/fp16/1-gpu-2-batch/nvsmi_numreqsample0_iter100_max500.out" \
													  "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/13B/fp16/1-gpu-4-batch/nvsmi_numreqsample0_iter100_max1000.out" \
								 					  "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/13B/fp16/1-gpu-4-batch/nvsmi_numreqsample0_iter100_max500.out" \
								 					  "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/13B/fp16/1-gpu-6-batch/nvsmi_numreqsample0_iter100_max500.out" \
								 --bmark_params       "13 1 1000" \
								 					  "13 1 500" \
													  "13 2 1000" \
													  "13 2 500" \
													  "13 4 1000" \
													  "13 4 500" \
													  "13 6 500" \
								 --plot_power_over_time \
								 --plot_filename "Llama13B_maxlen1000.png" \ 
								 --plot_sequence_length 1000

python gpu_batch_exp_plotting.py --bmark_output_paths "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/13B/fp16/1-gpu-1-batch/bmark_numreqsample0_iter100_max1000.out" \
								 					  "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/13B/fp16/1-gpu-1-batch/bmark_numreqsample0_iter100_max500.out" \
													  "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/13B/fp16/1-gpu-2-batch/bmark_numreqsample0_iter100_max1000.out" \
								 					  "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/13B/fp16/1-gpu-2-batch/bmark_numreqsample0_iter100_max500.out" \
													  "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/13B/fp16/1-gpu-4-batch/bmark_numreqsample0_iter100_max1000.out" \
								 					  "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/13B/fp16/1-gpu-4-batch/bmark_numreqsample0_iter100_max500.out" \
								 					  "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/13B/fp16/1-gpu-6-batch/bmark_numreqsample0_iter100_max500.out" \
								 --nvsmi_output_paths "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/13B/fp16/1-gpu-1-batch/nvsmi_numreqsample0_iter100_max1000.out" \
								 					  "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/13B/fp16/1-gpu-1-batch/nvsmi_numreqsample0_iter100_max500.out" \
													  "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/13B/fp16/1-gpu-2-batch/nvsmi_numreqsample0_iter100_max1000.out" \
								 					  "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/13B/fp16/1-gpu-2-batch/nvsmi_numreqsample0_iter100_max500.out" \
													  "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/13B/fp16/1-gpu-4-batch/nvsmi_numreqsample0_iter100_max1000.out" \
								 					  "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/13B/fp16/1-gpu-4-batch/nvsmi_numreqsample0_iter100_max500.out" \
								 					  "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/13B/fp16/1-gpu-6-batch/nvsmi_numreqsample0_iter100_max500.out" \
								 --bmark_params       "13 1 1000" \
								 					  "13 1 500" \
													  "13 2 1000" \
													  "13 2 500" \
													  "13 4 1000" \
													  "13 4 500" \
													  "13 6 500" \
								 --plot_power_over_time \
								 --plot_filename "Llama13B_maxlen500.png" \ 
								 --plot_sequence_length 500

# TODO: Finish OOM experiments for Llama7B max500
python gpu_batch_exp_plotting.py --bmark_output_paths ""
