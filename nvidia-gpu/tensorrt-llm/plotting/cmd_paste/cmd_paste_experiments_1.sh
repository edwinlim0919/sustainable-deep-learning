# it hurts so much



python3 gpu_batch_exp_plotting.py --bmark_output_paths "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/13B/fp16/1-gpu-1-batch/bmark_numreqsample0_iter100_max1000_a10040gb.out" \
													   "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/13B/fp16/1-gpu-2-batch/bmark_numreqsample0_iter100_max1000_a10040gb.out" \
													   "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/13B/fp16/1-gpu-4-batch/bmark_numreqsample0_iter100_max1000_a10040gb.out" \
													   "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/13B/fp16/1-gpu-6-batch/bmark_numreqsample0_iter100_max1000_a10040gb.out" \
													   "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/13B/fp16/1-gpu-8-batch/bmark_numreqsample0_iter100_max1000_a10040gb.out" \
								  --nvsmi_output_paths "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/13B/fp16/1-gpu-1-batch/nvsmi_numreqsample0_iter100_max1000_a10040gb.out" \
													   "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/13B/fp16/1-gpu-2-batch/nvsmi_numreqsample0_iter100_max1000_a10040gb.out" \
													   "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/13B/fp16/1-gpu-4-batch/nvsmi_numreqsample0_iter100_max1000_a10040gb.out" \
													   "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/13B/fp16/1-gpu-6-batch/nvsmi_numreqsample0_iter100_max1000_a10040gb.out" \
													   "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/13B/fp16/1-gpu-8-batch/nvsmi_numreqsample0_iter100_max1000_a10040gb.out" \
								  --bmark_params       "13B 1 1000 a10040gb" \
													   "13B 2 1000 a10040gb" \
													   "13B 4 1000 a10040gb" \
													   "13B 6 1000 a10040gb" \
													   "13B 8 1000 a10040gb" \
								  --bmark_param_groups "13B X 1000 a10040gb" \
								  --gpu_idx			   0 \
								  --plot_filename      "dev_testing_llama2_13b.png" \
								  --plot_name		   "Dev Testing Llama2 13B" \
								  --plot_tps_vs_tbt

python3 gpu_batch_exp_plotting.py --bmark_output_paths "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/13B/fp16/1-gpu-1-batch/bmark_numreqsample0_iter100_max1000_a10040gb.out" \
													   "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/13B/fp16/1-gpu-2-batch/bmark_numreqsample0_iter100_max1000_a10040gb.out" \
													   "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/13B/fp16/1-gpu-4-batch/bmark_numreqsample0_iter100_max1000_a10040gb.out" \
													   "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/13B/fp16/1-gpu-6-batch/bmark_numreqsample0_iter100_max1000_a10040gb.out" \
													   "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/13B/fp16/1-gpu-8-batch/bmark_numreqsample0_iter100_max1000_a10040gb.out" \
								  --nvsmi_output_paths "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/13B/fp16/1-gpu-1-batch/nvsmi_numreqsample0_iter100_max1000_a10040gb.out" \
													   "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/13B/fp16/1-gpu-2-batch/nvsmi_numreqsample0_iter100_max1000_a10040gb.out" \
													   "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/13B/fp16/1-gpu-4-batch/nvsmi_numreqsample0_iter100_max1000_a10040gb.out" \
													   "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/13B/fp16/1-gpu-6-batch/nvsmi_numreqsample0_iter100_max1000_a10040gb.out" \
													   "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/13B/fp16/1-gpu-8-batch/nvsmi_numreqsample0_iter100_max1000_a10040gb.out" \
								  --bmark_params       "13B 1 1000 a10040gb" \
													   "13B 2 1000 a10040gb" \
													   "13B 4 1000 a10040gb" \
													   "13B 6 1000 a10040gb" \
													   "13B 8 1000 a10040gb" \
								  --bmark_param_groups "13B X 1000 a10040gb" \
								  --gpu_idx			   0 \
								  --plot_filename      "dev_testing_llama2_13b.png" \
								  --plot_name		   "Dev Testing Llama2 13B" \
								  --plot_ept_vs_tbt

python3 gpu_batch_exp_plotting.py --bmark_output_paths "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/7B/fp16/1-gpu-1-batch/bmark_numreqsample0_iter100_max1000_a10040gb.out" \
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
								  --bmark_params       "7B 1 1000 a10040gb" \
								  					   "7B 4 1000 a10040gb" \
													   "7B 8 1000 a10040gb" \
													   "7B 12 1000 a10040gb" \
													   "7B 16 1000 a10040gb" \
													   "7B 20 1000 a10040gb" \
													   "7B 22 1000 a10040gb" \
													   "7B 1 1000 v10032gb" \
													   "7B 4 1000 v10032gb" \
													   "7B 8 1000 v10032gb" \
													   "7B 12 1000 v10032gb" \
													   "7B 14 1000 v10032gb" \
													   "13B 1 1000 a10040gb" \
													   "13B 2 1000 a10040gb" \
													   "13B 4 1000 a10040gb" \
													   "13B 6 1000 a10040gb" \
													   "13B 8 1000 a10040gb" \
													   "13B 1 1000 v10032gb" \
													   "13B 2 1000 v10032gb" \
													   "13B 4 1000 v10032gb" \
								  --bmark_param_groups "7B X 1000 a10040gb" \
								  					   "7B X 1000 v10032gb" \
													   "13B X 1000 a10040gb" \
													   "13B X 1000 v10032gb" \
								  --gpu_idx			   0 \
								  --plot_filename      "llama2_tbt_vs_ept_tradeoff.png" \
								  --plot_name		   "Llama2 TBT-EPT Tradeoff" \
								  --plot_tbt_vs_ept

# dev testing TCO breakdown
python3 gpu_batch_exp_plotting.py --bmark_output_paths "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/7B/fp16/1-gpu-1-batch/bmark_numreqsample0_iter100_max1000_a10040gb.out" \
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
								  --bmark_params       "7B 1 1000 a10040gb" \
								  					   "7B 4 1000 a10040gb" \
													   "7B 8 1000 a10040gb" \
													   "7B 12 1000 a10040gb" \
													   "7B 16 1000 a10040gb" \
													   "7B 20 1000 a10040gb" \
													   "7B 22 1000 a10040gb" \
								  --bmark_param_groups "7B X 1000 a10040gb" \
								  --gpu_idx			   0 \
								  --required_tps       1000000 \
								  --workload_duration_s 3600 \
								  --usd_per_kWh        0.165 \
								  --pue				   1.1 \
								  --gpu_lifetime_y     5 \
								  --plot_filename      "llama2_tco_breakdown.png" \
								  --plot_name		   "Llama2 TCO Breakdown" \
								  --plot_tco_breakdown

# TCO breakdown for max throughput
python3 gpu_batch_exp_plotting.py --bmark_output_paths "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/7B/fp16/1-gpu-1-batch/bmark_numreqsample0_iter100_max1000_a10040gb.out" \
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
								  --bmark_params       "7B 1 1000 a10040gb" \
								  					   "7B 4 1000 a10040gb" \
													   "7B 8 1000 a10040gb" \
													   "7B 12 1000 a10040gb" \
													   "7B 16 1000 a10040gb" \
													   "7B 20 1000 a10040gb" \
													   "7B 22 1000 a10040gb" \
													   "7B 1 1000 v10032gb" \
													   "7B 4 1000 v10032gb" \
													   "7B 8 1000 v10032gb" \
													   "7B 12 1000 v10032gb" \
													   "7B 14 1000 v10032gb" \
													   "13B 1 1000 a10040gb" \
													   "13B 2 1000 a10040gb" \
													   "13B 4 1000 a10040gb" \
													   "13B 6 1000 a10040gb" \
													   "13B 8 1000 a10040gb" \
													   "13B 1 1000 v10032gb" \
													   "13B 2 1000 v10032gb" \
													   "13B 4 1000 v10032gb" \
								  --bmark_param_groups "7B X 1000 a10040gb" \
								  					   "7B X 1000 v10032gb" \
													   "13B X 1000 a10040gb" \
													   "13B X 1000 v10032gb" \
								  --gpu_idx			   0 \
								  --required_tps       1000000 \
								  --workload_duration_s 3600 \
								  --usd_per_kWh        0.165 \
								  --pue				   1.1 \
								  --gpu_lifetime_y     5 \
								  --plot_filename      "llama2_tco_breakdown_max_throughput.png" \
								  --plot_name		   "Llama2 TCO Breakdown (Max. Throughput)" \
								  --plot_tco_breakdown

# TCF breakdown for 3 different single-gpu throughputs
python3 gpu_batch_exp_plotting.py --bmark_output_paths "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/7B/fp16/1-gpu-1-batch/bmark_numreqsample0_iter100_max1000_a10040gb.out" \
													   "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/7B/fp16/1-gpu-12-batch/bmark_numreqsample0_iter100_max1000_a10040gb.out" \
													   "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/7B/fp16/1-gpu-22-batch/bmark_numreqsample0_iter100_max1000_a10040gb.out" \
													   "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/7B/fp16/1-gpu-1-batch/bmark_numreqsample0_iter100_max1000_v10032gb.out" \
													   "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/7B/fp16/1-gpu-8-batch/bmark_numreqsample0_iter100_max1000_v10032gb.out" \
													   "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/7B/fp16/1-gpu-14-batch/bmark_numreqsample0_iter100_max1000_v10032gb.out" \
													   "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/13B/fp16/1-gpu-1-batch/bmark_numreqsample0_iter100_max1000_a10040gb.out" \
													   "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/13B/fp16/1-gpu-4-batch/bmark_numreqsample0_iter100_max1000_a10040gb.out" \
													   "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/13B/fp16/1-gpu-8-batch/bmark_numreqsample0_iter100_max1000_a10040gb.out" \
													   "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/13B/fp16/1-gpu-1-batch/bmark_numreqsample0_iter100_max1000_v10032gb.out" \
													   "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/13B/fp16/1-gpu-2-batch/bmark_numreqsample0_iter100_max1000_v10032gb.out" \
													   "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/13B/fp16/1-gpu-4-batch/bmark_numreqsample0_iter100_max1000_v10032gb.out" \
								  --nvsmi_output_paths "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/7B/fp16/1-gpu-1-batch/nvsmi_numreqsample0_iter100_max1000_a10040gb.out" \
													   "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/7B/fp16/1-gpu-12-batch/nvsmi_numreqsample0_iter100_max1000_a10040gb.out" \
													   "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/7B/fp16/1-gpu-22-batch/nvsmi_numreqsample0_iter100_max1000_a10040gb.out" \
													   "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/7B/fp16/1-gpu-1-batch/nvsmi_numreqsample0_iter100_max1000_v10032gb.out" \
													   "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/7B/fp16/1-gpu-8-batch/nvsmi_numreqsample0_iter100_max1000_v10032gb.out" \
													   "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/7B/fp16/1-gpu-14-batch/nvsmi_numreqsample0_iter100_max1000_v10032gb.out" \
													   "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/13B/fp16/1-gpu-1-batch/nvsmi_numreqsample0_iter100_max1000_a10040gb.out" \
													   "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/13B/fp16/1-gpu-4-batch/nvsmi_numreqsample0_iter100_max1000_a10040gb.out" \
													   "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/13B/fp16/1-gpu-8-batch/nvsmi_numreqsample0_iter100_max1000_a10040gb.out" \
													   "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/13B/fp16/1-gpu-1-batch/nvsmi_numreqsample0_iter100_max1000_v10032gb.out" \
													   "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/13B/fp16/1-gpu-2-batch/nvsmi_numreqsample0_iter100_max1000_v10032gb.out" \
													   "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/13B/fp16/1-gpu-4-batch/nvsmi_numreqsample0_iter100_max1000_v10032gb.out" \
								  --bmark_params       "7B 1 1000 a10040gb" \
													   "7B 12 1000 a10040gb" \
													   "7B 22 1000 a10040gb" \
													   "7B 1 1000 v10032gb" \
													   "7B 8 1000 v10032gb" \
													   "7B 14 1000 v10032gb" \
													   "13B 1 1000 a10040gb" \
													   "13B 4 1000 a10040gb" \
													   "13B 8 1000 a10040gb" \
													   "13B 1 1000 v10032gb" \
													   "13B 2 1000 v10032gb" \
													   "13B 4 1000 v10032gb" \
								  --bmark_param_groups "7B X 1000 a10040gb" \
								  					   "7B X 1000 v10032gb" \
													   "13B X 1000 a10040gb" \
													   "13B X 1000 v10032gb" \
								  --gpu_idx			   0 \
								  --required_tps       1000000 \
								  --workload_duration_s 3600 \
								  --gCO2eq_per_kWh     458 \
								  --pue				   1.1 \
								  --gpu_lifetime_y     5 \
								  --kgCO2eq_per_a10040gb 150 \
								  --kgCO2eq_per_v10032gb 111 \
								  --plot_filename      "llama2_tcf_breakdown_varied_throughput.png" \
								  --plot_name		   "Llama2 TCF Breakdown" \
								  --plot_tcf_breakdown\

# TCF breakdown for 3 different single-gpu throughputs
python3 gpu_batch_exp_plotting.py --bmark_output_paths "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/7B/fp16/1-gpu-1-batch/bmark_numreqsample0_iter100_max1000_a10040gb.out" \
													   "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/7B/fp16/1-gpu-12-batch/bmark_numreqsample0_iter100_max1000_a10040gb.out" \
													   "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/7B/fp16/1-gpu-22-batch/bmark_numreqsample0_iter100_max1000_a10040gb.out" \
													   "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/7B/fp16/1-gpu-1-batch/bmark_numreqsample0_iter100_max1000_v10032gb.out" \
													   "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/7B/fp16/1-gpu-8-batch/bmark_numreqsample0_iter100_max1000_v10032gb.out" \
													   "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/7B/fp16/1-gpu-14-batch/bmark_numreqsample0_iter100_max1000_v10032gb.out" \
													   "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/13B/fp16/1-gpu-1-batch/bmark_numreqsample0_iter100_max1000_a10040gb.out" \
													   "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/13B/fp16/1-gpu-4-batch/bmark_numreqsample0_iter100_max1000_a10040gb.out" \
													   "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/13B/fp16/1-gpu-8-batch/bmark_numreqsample0_iter100_max1000_a10040gb.out" \
													   "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/13B/fp16/1-gpu-1-batch/bmark_numreqsample0_iter100_max1000_v10032gb.out" \
													   "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/13B/fp16/1-gpu-2-batch/bmark_numreqsample0_iter100_max1000_v10032gb.out" \
													   "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/13B/fp16/1-gpu-4-batch/bmark_numreqsample0_iter100_max1000_v10032gb.out" \
								  --nvsmi_output_paths "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/7B/fp16/1-gpu-1-batch/nvsmi_numreqsample0_iter100_max1000_a10040gb.out" \
													   "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/7B/fp16/1-gpu-12-batch/nvsmi_numreqsample0_iter100_max1000_a10040gb.out" \
													   "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/7B/fp16/1-gpu-22-batch/nvsmi_numreqsample0_iter100_max1000_a10040gb.out" \
													   "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/7B/fp16/1-gpu-1-batch/nvsmi_numreqsample0_iter100_max1000_v10032gb.out" \
													   "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/7B/fp16/1-gpu-8-batch/nvsmi_numreqsample0_iter100_max1000_v10032gb.out" \
													   "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/7B/fp16/1-gpu-14-batch/nvsmi_numreqsample0_iter100_max1000_v10032gb.out" \
													   "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/13B/fp16/1-gpu-1-batch/nvsmi_numreqsample0_iter100_max1000_a10040gb.out" \
													   "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/13B/fp16/1-gpu-4-batch/nvsmi_numreqsample0_iter100_max1000_a10040gb.out" \
													   "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/13B/fp16/1-gpu-8-batch/nvsmi_numreqsample0_iter100_max1000_a10040gb.out" \
													   "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/13B/fp16/1-gpu-1-batch/nvsmi_numreqsample0_iter100_max1000_v10032gb.out" \
													   "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/13B/fp16/1-gpu-2-batch/nvsmi_numreqsample0_iter100_max1000_v10032gb.out" \
													   "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/13B/fp16/1-gpu-4-batch/nvsmi_numreqsample0_iter100_max1000_v10032gb.out" \
								  --bmark_params       "7B 1 1000 a10040gb" \
													   "7B 12 1000 a10040gb" \
													   "7B 22 1000 a10040gb" \
													   "7B 1 1000 v10032gb" \
													   "7B 8 1000 v10032gb" \
													   "7B 14 1000 v10032gb" \
													   "13B 1 1000 a10040gb" \
													   "13B 4 1000 a10040gb" \
													   "13B 8 1000 a10040gb" \
													   "13B 1 1000 v10032gb" \
													   "13B 2 1000 v10032gb" \
													   "13B 4 1000 v10032gb" \
								  --bmark_param_groups "7B X 1000 a10040gb" \
								  					   "7B X 1000 v10032gb" \
													   "13B X 1000 a10040gb" \
													   "13B X 1000 v10032gb" \
								  --gpu_idx			   0 \
								  --required_tps       1000000 \
								  --workload_duration_s 3600 \
								  --gCO2eq_per_kWh     24 \
								  --pue				   1.1 \
								  --gpu_lifetime_y     5 \
								  --kgCO2eq_per_a10040gb 150 \
								  --kgCO2eq_per_v10032gb 111 \
								  --plot_filename      "llama2_tcf_breakdown_varied_throughput.png" \
								  --plot_name		   "Llama2 TCF Breakdown" \
								  --plot_tcf_breakdown

# TCF breakdown for 3 different single-gpu throughputs
python3 gpu_batch_exp_plotting.py --bmark_output_paths "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/7B/fp16/1-gpu-1-batch/bmark_numreqsample0_iter100_max1000_a10040gb.out" \
													   "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/7B/fp16/1-gpu-12-batch/bmark_numreqsample0_iter100_max1000_a10040gb.out" \
													   "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/7B/fp16/1-gpu-22-batch/bmark_numreqsample0_iter100_max1000_a10040gb.out" \
													   "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/7B/fp16/1-gpu-1-batch/bmark_numreqsample0_iter100_max1000_v10032gb.out" \
													   "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/7B/fp16/1-gpu-8-batch/bmark_numreqsample0_iter100_max1000_v10032gb.out" \
													   "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/7B/fp16/1-gpu-14-batch/bmark_numreqsample0_iter100_max1000_v10032gb.out" \
													   "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/13B/fp16/1-gpu-1-batch/bmark_numreqsample0_iter100_max1000_a10040gb.out" \
													   "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/13B/fp16/1-gpu-4-batch/bmark_numreqsample0_iter100_max1000_a10040gb.out" \
													   "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/13B/fp16/1-gpu-8-batch/bmark_numreqsample0_iter100_max1000_a10040gb.out" \
													   "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/13B/fp16/1-gpu-1-batch/bmark_numreqsample0_iter100_max1000_v10032gb.out" \
													   "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/13B/fp16/1-gpu-2-batch/bmark_numreqsample0_iter100_max1000_v10032gb.out" \
													   "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/13B/fp16/1-gpu-4-batch/bmark_numreqsample0_iter100_max1000_v10032gb.out" \
								  --nvsmi_output_paths "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/7B/fp16/1-gpu-1-batch/nvsmi_numreqsample0_iter100_max1000_a10040gb.out" \
													   "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/7B/fp16/1-gpu-12-batch/nvsmi_numreqsample0_iter100_max1000_a10040gb.out" \
													   "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/7B/fp16/1-gpu-22-batch/nvsmi_numreqsample0_iter100_max1000_a10040gb.out" \
													   "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/7B/fp16/1-gpu-1-batch/nvsmi_numreqsample0_iter100_max1000_v10032gb.out" \
													   "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/7B/fp16/1-gpu-8-batch/nvsmi_numreqsample0_iter100_max1000_v10032gb.out" \
													   "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/7B/fp16/1-gpu-14-batch/nvsmi_numreqsample0_iter100_max1000_v10032gb.out" \
													   "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/13B/fp16/1-gpu-1-batch/nvsmi_numreqsample0_iter100_max1000_a10040gb.out" \
													   "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/13B/fp16/1-gpu-4-batch/nvsmi_numreqsample0_iter100_max1000_a10040gb.out" \
													   "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/13B/fp16/1-gpu-8-batch/nvsmi_numreqsample0_iter100_max1000_a10040gb.out" \
													   "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/13B/fp16/1-gpu-1-batch/nvsmi_numreqsample0_iter100_max1000_v10032gb.out" \
													   "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/13B/fp16/1-gpu-2-batch/nvsmi_numreqsample0_iter100_max1000_v10032gb.out" \
													   "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/13B/fp16/1-gpu-4-batch/nvsmi_numreqsample0_iter100_max1000_v10032gb.out" \
								  --bmark_params       "7B 1 1000 a10040gb" \
													   "7B 12 1000 a10040gb" \
													   "7B 22 1000 a10040gb" \
													   "7B 1 1000 v10032gb" \
													   "7B 8 1000 v10032gb" \
													   "7B 14 1000 v10032gb" \
													   "13B 1 1000 a10040gb" \
													   "13B 4 1000 a10040gb" \
													   "13B 8 1000 a10040gb" \
													   "13B 1 1000 v10032gb" \
													   "13B 2 1000 v10032gb" \
													   "13B 4 1000 v10032gb" \
								  --bmark_param_groups "7B X 1000 a10040gb" \
								  					   "7B X 1000 v10032gb" \
													   "13B X 1000 a10040gb" \
													   "13B X 1000 v10032gb" \
								  --gpu_idx			   0 \
								  --required_tps       1000000 \
								  --workload_duration_s 3600 \
								  --gCO2eq_per_kWh     24 \
								  --pue				   1.1 \
								  --gpu_lifetime_y     5 \
								  --kgCO2eq_per_a10040gb 150 \
								  --kgCO2eq_per_v10032gb 0 \
								  --plot_filename      "llama2_tcf_breakdown_varied_throughput_second_life.png" \
								  --plot_name		   "Llama2 TCF Breakdown w/ Second Life V100s" \
								  --plot_tcf_breakdown

# TCO breakdown for 3 different single-gpu throughputs
python3 gpu_batch_exp_plotting.py --bmark_output_paths "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/7B/fp16/1-gpu-1-batch/bmark_numreqsample0_iter100_max1000_a10040gb.out" \
													   "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/7B/fp16/1-gpu-12-batch/bmark_numreqsample0_iter100_max1000_a10040gb.out" \
													   "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/7B/fp16/1-gpu-22-batch/bmark_numreqsample0_iter100_max1000_a10040gb.out" \
													   "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/7B/fp16/1-gpu-1-batch/bmark_numreqsample0_iter100_max1000_v10032gb.out" \
													   "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/7B/fp16/1-gpu-8-batch/bmark_numreqsample0_iter100_max1000_v10032gb.out" \
													   "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/7B/fp16/1-gpu-14-batch/bmark_numreqsample0_iter100_max1000_v10032gb.out" \
													   "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/13B/fp16/1-gpu-1-batch/bmark_numreqsample0_iter100_max1000_a10040gb.out" \
													   "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/13B/fp16/1-gpu-4-batch/bmark_numreqsample0_iter100_max1000_a10040gb.out" \
													   "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/13B/fp16/1-gpu-8-batch/bmark_numreqsample0_iter100_max1000_a10040gb.out" \
													   "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/13B/fp16/1-gpu-1-batch/bmark_numreqsample0_iter100_max1000_v10032gb.out" \
													   "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/13B/fp16/1-gpu-2-batch/bmark_numreqsample0_iter100_max1000_v10032gb.out" \
													   "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/13B/fp16/1-gpu-4-batch/bmark_numreqsample0_iter100_max1000_v10032gb.out" \
								  --nvsmi_output_paths "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/7B/fp16/1-gpu-1-batch/nvsmi_numreqsample0_iter100_max1000_a10040gb.out" \
													   "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/7B/fp16/1-gpu-12-batch/nvsmi_numreqsample0_iter100_max1000_a10040gb.out" \
													   "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/7B/fp16/1-gpu-22-batch/nvsmi_numreqsample0_iter100_max1000_a10040gb.out" \
													   "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/7B/fp16/1-gpu-1-batch/nvsmi_numreqsample0_iter100_max1000_v10032gb.out" \
													   "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/7B/fp16/1-gpu-8-batch/nvsmi_numreqsample0_iter100_max1000_v10032gb.out" \
													   "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/7B/fp16/1-gpu-14-batch/nvsmi_numreqsample0_iter100_max1000_v10032gb.out" \
													   "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/13B/fp16/1-gpu-1-batch/nvsmi_numreqsample0_iter100_max1000_a10040gb.out" \
													   "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/13B/fp16/1-gpu-4-batch/nvsmi_numreqsample0_iter100_max1000_a10040gb.out" \
													   "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/13B/fp16/1-gpu-8-batch/nvsmi_numreqsample0_iter100_max1000_a10040gb.out" \
													   "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/13B/fp16/1-gpu-1-batch/nvsmi_numreqsample0_iter100_max1000_v10032gb.out" \
													   "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/13B/fp16/1-gpu-2-batch/nvsmi_numreqsample0_iter100_max1000_v10032gb.out" \
													   "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/13B/fp16/1-gpu-4-batch/nvsmi_numreqsample0_iter100_max1000_v10032gb.out" \
								  --bmark_params       "7B 1 1000 a10040gb" \
													   "7B 12 1000 a10040gb" \
													   "7B 22 1000 a10040gb" \
													   "7B 1 1000 v10032gb" \
													   "7B 8 1000 v10032gb" \
													   "7B 14 1000 v10032gb" \
													   "13B 1 1000 a10040gb" \
													   "13B 4 1000 a10040gb" \
													   "13B 8 1000 a10040gb" \
													   "13B 1 1000 v10032gb" \
													   "13B 2 1000 v10032gb" \
													   "13B 4 1000 v10032gb" \
								  --bmark_param_groups "7B X 1000 a10040gb" \
								  					   "7B X 1000 v10032gb" \
													   "13B X 1000 a10040gb" \
													   "13B X 1000 v10032gb" \
								  --gpu_idx			   0 \
								  --required_tps       1000000 \
								  --workload_duration_s 3600 \
								  --usd_per_kWh        0.165 \
								  --pue				   1.1 \
								  --gpu_lifetime_y     5 \
								  --usd_per_a10040gb   9000 \
								  --usd_per_v10032gb   0 \
								  --plot_filename      "llama2_tco_breakeven_varied_throughput.png" \
								  --plot_name		   "Llama2 TCO Breakeven Points w/ Second Life V100s" \
								  --plot_tco_breakeven

python3 gpu_batch_exp_plotting.py --bmark_output_paths "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/falcon/40B/bf16/tp4-pp1-batch1/bmark_numreqsample0_iter100_max1000_a10040gb.out" \
													   "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/falcon/40B/bf16/tp4-pp1-batch8/bmark_numreqsample0_iter100_max1000_a10040gb.out" \
													   "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/falcon/40B/bf16/tp4-pp1-batch16/bmark_numreqsample0_iter100_max1000_a10040gb.out" \
													   "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/falcon/40B/bf16/tp4-pp1-batch24/bmark_numreqsample0_iter100_max1000_a10040gb.out" \
													   "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/falcon/40B/bf16/tp4-pp1-batch32/bmark_numreqsample0_iter100_max1000_a10040gb.out" \
													   "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/falcon/40B/bf16/tp4-pp1-batch40/bmark_numreqsample0_iter100_max1000_a10040gb.out" \
													   "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/falcon/40B/bf16/tp4-pp1-batch48/bmark_numreqsample0_iter100_max1000_a10040gb.out" \
													   "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/falcon/40B/bf16/tp4-pp1-batch56/bmark_numreqsample0_iter100_max1000_a10040gb.out" \
													   "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/falcon/40B/bf16/tp4-pp1-batch64/bmark_numreqsample0_iter100_max1000_a10040gb.out" \
													   "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/falcon/40B/bf16/tp4-pp1-batch80/bmark_numreqsample0_iter100_max1000_a10040gb.out" \
													   "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/falcon/40B/bf16/tp4-pp1-batch96/bmark_numreqsample0_iter100_max1000_a10040gb.out" \
													   "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/falcon/40B/bf16/tp4-pp1-batch128/bmark_numreqsample0_iter100_max1000_a10040gb.out" \
													   "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/falcon/40B/bf16/tp4-pp1-batch160/bmark_numreqsample0_iter100_max1000_a10040gb.out" \
													   "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/falcon/40B/bf16/tp4-pp1-batch192/bmark_numreqsample0_iter100_max1000_a10040gb.out" \
													   "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/falcon/40B/bf16/tp4-pp1-batch256/bmark_numreqsample0_iter100_max1000_a10040gb.out" \
													   "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/falcon/40B/bf16/tp4-pp1-batch384/bmark_numreqsample0_iter100_max1000_a10040gb.out" \
								  --nvsmi_output_paths "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/falcon/40B/bf16/tp4-pp1-batch1/nvsmi_numreqsample0_iter100_max1000_a10040gb.out" \
													   "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/falcon/40B/bf16/tp4-pp1-batch8/nvsmi_numreqsample0_iter100_max1000_a10040gb.out" \
													   "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/falcon/40B/bf16/tp4-pp1-batch16/nvsmi_numreqsample0_iter100_max1000_a10040gb.out" \
													   "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/falcon/40B/bf16/tp4-pp1-batch24/nvsmi_numreqsample0_iter100_max1000_a10040gb.out" \
													   "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/falcon/40B/bf16/tp4-pp1-batch32/nvsmi_numreqsample0_iter100_max1000_a10040gb.out" \
													   "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/falcon/40B/bf16/tp4-pp1-batch40/nvsmi_numreqsample0_iter100_max1000_a10040gb.out" \
													   "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/falcon/40B/bf16/tp4-pp1-batch48/nvsmi_numreqsample0_iter100_max1000_a10040gb.out" \
													   "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/falcon/40B/bf16/tp4-pp1-batch56/nvsmi_numreqsample0_iter100_max1000_a10040gb.out" \
													   "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/falcon/40B/bf16/tp4-pp1-batch64/nvsmi_numreqsample0_iter100_max1000_a10040gb.out" \
													   "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/falcon/40B/bf16/tp4-pp1-batch80/nvsmi_numreqsample0_iter100_max1000_a10040gb.out" \
													   "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/falcon/40B/bf16/tp4-pp1-batch96/nvsmi_numreqsample0_iter100_max1000_a10040gb.out" \
													   "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/falcon/40B/bf16/tp4-pp1-batch128/nvsmi_numreqsample0_iter100_max1000_a10040gb.out" \
													   "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/falcon/40B/bf16/tp4-pp1-batch160/nvsmi_numreqsample0_iter100_max1000_a10040gb.out" \
													   "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/falcon/40B/bf16/tp4-pp1-batch192/nvsmi_numreqsample0_iter100_max1000_a10040gb.out" \
													   "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/falcon/40B/bf16/tp4-pp1-batch256/nvsmi_numreqsample0_iter100_max1000_a10040gb.out" \
													   "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/falcon/40B/bf16/tp4-pp1-batch384/nvsmi_numreqsample0_iter100_max1000_a10040gb.out" \
								  --bmark_params       "40B 1 1000 a10040gb" \
													   "40B 8 1000 a10040gb" \
													   "40B 16 1000 a10040gb" \
													   "40B 24 1000 a10040gb" \
													   "40B 32 1000 a10040gb" \
													   "40B 40 1000 a10040gb" \
													   "40B 48 1000 a10040gb" \
													   "40B 56 1000 a10040gb" \
													   "40B 64 1000 a10040gb" \
													   "40B 80 1000 a10040gb" \
													   "40B 96 1000 a10040gb" \
													   "40B 128 1000 a10040gb" \
													   "40B 160 1000 a10040gb" \
													   "40B 192 1000 a10040gb" \
													   "40B 256 1000 a10040gb" \
													   "40B 384 1000 a10040gb" \
								  --bmark_param_groups "40B X 1000 a10040gb" \
								  --gpu_idxs		   0 \
								                       1 \
													   2 \
													   3 \
								  --plot_filename      "falcon_tps_vs_tbt_tradeoff.png" \
								  --plot_name		   "Falcon 40B TPS-TBT Tradeoff" \
								  --tbt_slo      	   0.1 \
								  --plot_tps_vs_tbt

python3 gpu_batch_exp_plotting.py --bmark_output_paths "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/7B/fp16/1-gpu-1-batch/bmark_numreqsample0_iter100_max1000_a10040gb.out" \
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
								  --bmark_params       "7B 1 1000 a10040gb" \
								  					   "7B 4 1000 a10040gb" \
													   "7B 8 1000 a10040gb" \
													   "7B 12 1000 a10040gb" \
													   "7B 16 1000 a10040gb" \
													   "7B 20 1000 a10040gb" \
													   "7B 22 1000 a10040gb" \
													   "7B 1 1000 v10032gb" \
													   "7B 4 1000 v10032gb" \
													   "7B 8 1000 v10032gb" \
													   "7B 12 1000 v10032gb" \
													   "7B 14 1000 v10032gb" \
													   "13B 1 1000 a10040gb" \
													   "13B 2 1000 a10040gb" \
													   "13B 4 1000 a10040gb" \
													   "13B 6 1000 a10040gb" \
													   "13B 8 1000 a10040gb" \
													   "13B 1 1000 v10032gb" \
													   "13B 2 1000 v10032gb" \
													   "13B 4 1000 v10032gb" \
								  --bmark_param_groups "7B X 1000 a10040gb" \
								  					   "7B X 1000 v10032gb" \
													   "13B X 1000 a10040gb" \
													   "13B X 1000 v10032gb" \
								  --gpu_idx			   0 \
								  --plot_filename      "llama2_tps_vs_tbt_tradeoff.png" \
								  --plot_name		   "Llama2 TPS-TBT Tradeoff" \
								  --tbt_slo      	   0.1 \
								  --plot_tps_vs_tbt

python3 gpu_batch_exp_plotting.py --bmark_output_paths "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/falcon/40B/bf16/tp4-pp1-batch1/bmark_numreqsample0_iter100_max1000_a10040gb.out" \
													   "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/falcon/40B/bf16/tp4-pp1-batch8/bmark_numreqsample0_iter100_max1000_a10040gb.out" \
													   "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/falcon/40B/bf16/tp4-pp1-batch16/bmark_numreqsample0_iter100_max1000_a10040gb.out" \
													   "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/falcon/40B/bf16/tp4-pp1-batch24/bmark_numreqsample0_iter100_max1000_a10040gb.out" \
													   "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/falcon/40B/bf16/tp4-pp1-batch32/bmark_numreqsample0_iter100_max1000_a10040gb.out" \
													   "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/falcon/40B/bf16/tp4-pp1-batch40/bmark_numreqsample0_iter100_max1000_a10040gb.out" \
													   "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/falcon/40B/bf16/tp4-pp1-batch48/bmark_numreqsample0_iter100_max1000_a10040gb.out" \
													   "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/falcon/40B/bf16/tp4-pp1-batch56/bmark_numreqsample0_iter100_max1000_a10040gb.out" \
													   "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/falcon/40B/bf16/tp4-pp1-batch64/bmark_numreqsample0_iter100_max1000_a10040gb.out" \
													   "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/falcon/40B/bf16/tp4-pp1-batch80/bmark_numreqsample0_iter100_max1000_a10040gb.out" \
													   "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/falcon/40B/bf16/tp4-pp1-batch96/bmark_numreqsample0_iter100_max1000_a10040gb.out" \
													   "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/falcon/40B/bf16/tp4-pp1-batch128/bmark_numreqsample0_iter100_max1000_a10040gb.out" \
													   "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/falcon/40B/bf16/tp4-pp1-batch160/bmark_numreqsample0_iter100_max1000_a10040gb.out" \
													   "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/falcon/40B/bf16/tp4-pp1-batch192/bmark_numreqsample0_iter100_max1000_a10040gb.out" \
													   "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/falcon/40B/bf16/tp4-pp1-batch256/bmark_numreqsample0_iter100_max1000_a10040gb.out" \
													   "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/falcon/40B/bf16/tp4-pp1-batch384/bmark_numreqsample0_iter100_max1000_a10040gb.out" \
								  --nvsmi_output_paths "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/falcon/40B/bf16/tp4-pp1-batch1/nvsmi_numreqsample0_iter100_max1000_a10040gb.out" \
													   "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/falcon/40B/bf16/tp4-pp1-batch8/nvsmi_numreqsample0_iter100_max1000_a10040gb.out" \
													   "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/falcon/40B/bf16/tp4-pp1-batch16/nvsmi_numreqsample0_iter100_max1000_a10040gb.out" \
													   "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/falcon/40B/bf16/tp4-pp1-batch24/nvsmi_numreqsample0_iter100_max1000_a10040gb.out" \
													   "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/falcon/40B/bf16/tp4-pp1-batch32/nvsmi_numreqsample0_iter100_max1000_a10040gb.out" \
													   "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/falcon/40B/bf16/tp4-pp1-batch40/nvsmi_numreqsample0_iter100_max1000_a10040gb.out" \
													   "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/falcon/40B/bf16/tp4-pp1-batch48/nvsmi_numreqsample0_iter100_max1000_a10040gb.out" \
													   "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/falcon/40B/bf16/tp4-pp1-batch56/nvsmi_numreqsample0_iter100_max1000_a10040gb.out" \
													   "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/falcon/40B/bf16/tp4-pp1-batch64/nvsmi_numreqsample0_iter100_max1000_a10040gb.out" \
													   "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/falcon/40B/bf16/tp4-pp1-batch80/nvsmi_numreqsample0_iter100_max1000_a10040gb.out" \
													   "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/falcon/40B/bf16/tp4-pp1-batch96/nvsmi_numreqsample0_iter100_max1000_a10040gb.out" \
													   "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/falcon/40B/bf16/tp4-pp1-batch128/nvsmi_numreqsample0_iter100_max1000_a10040gb.out" \
													   "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/falcon/40B/bf16/tp4-pp1-batch160/nvsmi_numreqsample0_iter100_max1000_a10040gb.out" \
													   "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/falcon/40B/bf16/tp4-pp1-batch192/nvsmi_numreqsample0_iter100_max1000_a10040gb.out" \
													   "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/falcon/40B/bf16/tp4-pp1-batch256/nvsmi_numreqsample0_iter100_max1000_a10040gb.out" \
													   "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/falcon/40B/bf16/tp4-pp1-batch384/nvsmi_numreqsample0_iter100_max1000_a10040gb.out" \
								  --bmark_params       "40B 1 1000 a10040gb" \
													   "40B 8 1000 a10040gb" \
													   "40B 16 1000 a10040gb" \
													   "40B 24 1000 a10040gb" \
													   "40B 32 1000 a10040gb" \
													   "40B 40 1000 a10040gb" \
													   "40B 48 1000 a10040gb" \
													   "40B 56 1000 a10040gb" \
													   "40B 64 1000 a10040gb" \
													   "40B 80 1000 a10040gb" \
													   "40B 96 1000 a10040gb" \
													   "40B 128 1000 a10040gb" \
													   "40B 160 1000 a10040gb" \
													   "40B 192 1000 a10040gb" \
													   "40B 256 1000 a10040gb" \
													   "40B 384 1000 a10040gb" \
								  --bmark_param_groups "40B X 1000 a10040gb" \
								  --gpu_idxs		   0 \
								                       1 \
													   2 \
													   3 \
								  --plot_filename      "falcon_tbt_vs_ept_tradeoff.png" \
								  --plot_name		   "Falcon 40B TBT-EPT Tradeoff" \
								  --plot_tbt_vs_ept

# TCO breakdown for 3 different single-gpu throughputs
python3 gpu_batch_exp_plotting.py --bmark_output_paths  "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/7B/fp16/1-gpu-1-batch/bmark_numreqsample0_iter100_max1000_a10040gb.out" \
													    "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/7B/fp16/1-gpu-12-batch/bmark_numreqsample0_iter100_max1000_a10040gb.out" \
													    "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/7B/fp16/1-gpu-22-batch/bmark_numreqsample0_iter100_max1000_a10040gb.out" \
													    "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/7B/fp16/1-gpu-1-batch/bmark_numreqsample0_iter100_max1000_v10032gb.out" \
													    "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/7B/fp16/1-gpu-8-batch/bmark_numreqsample0_iter100_max1000_v10032gb.out" \
													    "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/7B/fp16/1-gpu-14-batch/bmark_numreqsample0_iter100_max1000_v10032gb.out" \
													    "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/13B/fp16/1-gpu-1-batch/bmark_numreqsample0_iter100_max1000_a10040gb.out" \
													    "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/13B/fp16/1-gpu-4-batch/bmark_numreqsample0_iter100_max1000_a10040gb.out" \
													    "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/13B/fp16/1-gpu-8-batch/bmark_numreqsample0_iter100_max1000_a10040gb.out" \
													    "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/13B/fp16/1-gpu-1-batch/bmark_numreqsample0_iter100_max1000_v10032gb.out" \
													    "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/13B/fp16/1-gpu-2-batch/bmark_numreqsample0_iter100_max1000_v10032gb.out" \
													    "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/13B/fp16/1-gpu-4-batch/bmark_numreqsample0_iter100_max1000_v10032gb.out" \
								  --nvsmi_output_paths  "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/7B/fp16/1-gpu-1-batch/nvsmi_numreqsample0_iter100_max1000_a10040gb.out" \
													    "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/7B/fp16/1-gpu-12-batch/nvsmi_numreqsample0_iter100_max1000_a10040gb.out" \
													    "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/7B/fp16/1-gpu-22-batch/nvsmi_numreqsample0_iter100_max1000_a10040gb.out" \
													    "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/7B/fp16/1-gpu-1-batch/nvsmi_numreqsample0_iter100_max1000_v10032gb.out" \
													    "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/7B/fp16/1-gpu-8-batch/nvsmi_numreqsample0_iter100_max1000_v10032gb.out" \
													    "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/7B/fp16/1-gpu-14-batch/nvsmi_numreqsample0_iter100_max1000_v10032gb.out" \
													    "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/13B/fp16/1-gpu-1-batch/nvsmi_numreqsample0_iter100_max1000_a10040gb.out" \
													    "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/13B/fp16/1-gpu-4-batch/nvsmi_numreqsample0_iter100_max1000_a10040gb.out" \
													    "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/13B/fp16/1-gpu-8-batch/nvsmi_numreqsample0_iter100_max1000_a10040gb.out" \
													    "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/13B/fp16/1-gpu-1-batch/nvsmi_numreqsample0_iter100_max1000_v10032gb.out" \
													    "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/13B/fp16/1-gpu-2-batch/nvsmi_numreqsample0_iter100_max1000_v10032gb.out" \
													    "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/13B/fp16/1-gpu-4-batch/nvsmi_numreqsample0_iter100_max1000_v10032gb.out" \
								  --bmark_params        "7B 1 1000 a10040gb" \
													    "7B 12 1000 a10040gb" \
													    "7B 22 1000 a10040gb" \
													    "7B 1 1000 v10032gb" \
													    "7B 8 1000 v10032gb" \
													    "7B 14 1000 v10032gb" \
													    "13B 1 1000 a10040gb" \
													    "13B 4 1000 a10040gb" \
													    "13B 8 1000 a10040gb" \
													    "13B 1 1000 v10032gb" \
													    "13B 2 1000 v10032gb" \
													    "13B 4 1000 v10032gb" \
								  --bmark_param_groups  "7B X 1000 a10040gb" \
								  					    "7B X 1000 v10032gb" \
													    "13B X 1000 a10040gb" \
													    "13B X 1000 v10032gb" \
								  --gpu_idxs		    0 \
								  --required_tps        1000000 \
								  --workload_duration_s 3600 \
								  --usd_per_kWh         0.165 \
								  --pue				    1.1 \
								  --server_lifetime_y   5 \
								  --pkg_power_load      "pkg_power_50" \
								  --ram_power_load      "ram_power_50" \
								  --plot_filename       "llama2_tco_breakdown_varied_throughput.png" \
								  --plot_name		    "Llama2 TCO Breakdown" \
								  --plot_tco_breakdown

# TCO breakdown for 3 different single-gpu throughputs
python3 gpu_batch_exp_plotting.py --bmark_output_paths  "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/7B/fp16/1-gpu-1-batch/bmark_numreqsample0_iter100_max1000_a10040gb.out" \
													    "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/7B/fp16/1-gpu-12-batch/bmark_numreqsample0_iter100_max1000_a10040gb.out" \
													    "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/7B/fp16/1-gpu-22-batch/bmark_numreqsample0_iter100_max1000_a10040gb.out" \
													    "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/7B/fp16/1-gpu-1-batch/bmark_numreqsample0_iter100_max1000_v10032gb.out" \
													    "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/7B/fp16/1-gpu-8-batch/bmark_numreqsample0_iter100_max1000_v10032gb.out" \
													    "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/7B/fp16/1-gpu-14-batch/bmark_numreqsample0_iter100_max1000_v10032gb.out" \
													    "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/13B/fp16/1-gpu-1-batch/bmark_numreqsample0_iter100_max1000_a10040gb.out" \
													    "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/13B/fp16/1-gpu-4-batch/bmark_numreqsample0_iter100_max1000_a10040gb.out" \
													    "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/13B/fp16/1-gpu-8-batch/bmark_numreqsample0_iter100_max1000_a10040gb.out" \
													    "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/13B/fp16/1-gpu-1-batch/bmark_numreqsample0_iter100_max1000_v10032gb.out" \
													    "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/13B/fp16/1-gpu-2-batch/bmark_numreqsample0_iter100_max1000_v10032gb.out" \
													    "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/13B/fp16/1-gpu-4-batch/bmark_numreqsample0_iter100_max1000_v10032gb.out" \
								  --nvsmi_output_paths  "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/7B/fp16/1-gpu-1-batch/nvsmi_numreqsample0_iter100_max1000_a10040gb.out" \
													    "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/7B/fp16/1-gpu-12-batch/nvsmi_numreqsample0_iter100_max1000_a10040gb.out" \
													    "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/7B/fp16/1-gpu-22-batch/nvsmi_numreqsample0_iter100_max1000_a10040gb.out" \
													    "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/7B/fp16/1-gpu-1-batch/nvsmi_numreqsample0_iter100_max1000_v10032gb.out" \
													    "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/7B/fp16/1-gpu-8-batch/nvsmi_numreqsample0_iter100_max1000_v10032gb.out" \
													    "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/7B/fp16/1-gpu-14-batch/nvsmi_numreqsample0_iter100_max1000_v10032gb.out" \
													    "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/13B/fp16/1-gpu-1-batch/nvsmi_numreqsample0_iter100_max1000_a10040gb.out" \
													    "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/13B/fp16/1-gpu-4-batch/nvsmi_numreqsample0_iter100_max1000_a10040gb.out" \
													    "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/13B/fp16/1-gpu-8-batch/nvsmi_numreqsample0_iter100_max1000_a10040gb.out" \
													    "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/13B/fp16/1-gpu-1-batch/nvsmi_numreqsample0_iter100_max1000_v10032gb.out" \
													    "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/13B/fp16/1-gpu-2-batch/nvsmi_numreqsample0_iter100_max1000_v10032gb.out" \
													    "/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt-llm/outputs/llama/13B/fp16/1-gpu-4-batch/nvsmi_numreqsample0_iter100_max1000_v10032gb.out" \
								  --bmark_params        "7B 1 1000 a10040gb" \
													    "7B 12 1000 a10040gb" \
													    "7B 22 1000 a10040gb" \
													    "7B 1 1000 v10032gb" \
													    "7B 8 1000 v10032gb" \
													    "7B 14 1000 v10032gb" \
													    "13B 1 1000 a10040gb" \
													    "13B 4 1000 a10040gb" \
													    "13B 8 1000 a10040gb" \
													    "13B 1 1000 v10032gb" \
													    "13B 2 1000 v10032gb" \
													    "13B 4 1000 v10032gb" \
								  --bmark_param_groups  "7B X 1000 a10040gb" \
								  					    "7B X 1000 v10032gb" \
													    "13B X 1000 a10040gb" \
													    "13B X 1000 v10032gb" \
								  --gpu_idxs		    0 \
								  --required_tps        1000000 \
								  --workload_duration_s 3600 \
								  --usd_per_kWh         0.165 \
								  --pue				    1.1 \
								  --server_lifetime_y   5 \
								  --second_life         \
								  --pkg_power_load      "pkg_power_50" \
								  --ram_power_load      "ram_power_50" \
								  --plot_filename       "llama2_tco_breakdown_varied_throughput_second_life.png" \
								  --plot_name		    "Llama2 TCO Breakdown w/ Second Life V100s" \
								  --plot_tco_breakdown


