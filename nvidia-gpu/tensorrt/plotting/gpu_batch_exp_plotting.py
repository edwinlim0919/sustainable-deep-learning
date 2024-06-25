import argparse
import matplotlib.pyplot as plt
import gpu_batch_exp_utils
from collections import defaultdict

def calculate_average_power_usage(bmark_info, nvsmi_info):
    power_usage_per_batch = defaultdict(list)
    
    for iteration, batch_data in bmark_info.items():
        start_time = batch_data['batch_start_time']
        end_time = batch_data['batch_end_time']
        
        relevant_nvsmi_data = [data for data in nvsmi_info if start_time <= data['timestamp_raw'] <= end_time]
        
        if relevant_nvsmi_data:
            avg_power = sum(data[0]['curr_power_usage'] for data in relevant_nvsmi_data) / len(relevant_nvsmi_data)
            power_usage_per_batch[iteration].append(avg_power)
    
    return power_usage_per_batch

def plot_batch_size_vs_power(bmark_output_path, nvsmi_output_path):
    bmark_info = gpu_batch_exp_utils.parse_bmark_output(bmark_output_path)
    nvsmi_info = gpu_batch_exp_utils.parse_nvsmi_output(nvsmi_output_path)
    
    power_usage_per_batch = calculate_average_power_usage(bmark_info, nvsmi_info)
    
    batch_sizes = sorted(power_usage_per_batch.keys())
    avg_power_usage = [sum(power_usage_per_batch[batch]) / len(power_usage_per_batch[batch]) for batch in batch_sizes]
    
    plt.figure(figsize=(10, 6))
    plt.plot(batch_sizes, avg_power_usage, marker='o')
    plt.xlabel('Batch Size')
    plt.ylabel('Average Power Usage (W)')
    plt.title('Batch Size vs. Average Power Usage')
    plt.grid(True)
    plt.savefig('batch_size_vs_power_usage.png')
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot batch size vs. average power usage")
    parser.add_argument("bmark_output", help="Path to the benchmark output file")
    parser.add_argument("nvsmi_output", help="Path to the NVSMI output file")
    args = parser.parse_args()

    plot_batch_size_vs_power(args.bmark_output, args.nvsmi_output)
