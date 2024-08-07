import argparse
import re
import ast
import matplotlib.pyplot as plt
from pathlib import Path

def parse_bmark_output(bmark_output_path):
    print(f'parse_bmark_output: {bmark_output_path}')
    with open(bmark_output_path, 'r') as f:
        bmark_output_lines = f.readlines()

    num_iterations_line = bmark_output_lines[0]
    num_iterations = int(num_iterations_line.split()[-1])
    batch_size_line = bmark_output_lines[1]
    batch_size = int(batch_size_line.split()[-1])
    bmark_info = {}
    
    for line in bmark_output_lines[2:]:
        if 'iteration' in line:
            curr_iteration = int(line.strip().split()[-1].split('/')[0])
            bmark_info[curr_iteration] = {}

        if 'start_time' in line:
            bmark_info[curr_iteration]['batch_start_time'] = float(line.strip().split()[-1])

        if 'end_time' in line:
            bmark_info[curr_iteration]['batch_end_time'] = float(line.strip().split()[-1])

    return bmark_info


temp_celsius_pattern = r'(\d+)C'
power_usage_pattern = r'(\d+)W / (\d+)W'
memory_usage_pattern = r'(\d+)MiB / (\d+)MiB'
gpu_utilization_pattern = r'(\d+)%'

def parse_nvsmi_output(nvsmi_output_path):
    print(f'parse_nvsmi_output: {nvsmi_output_path}')
    with open(nvsmi_output_path, 'r') as f:
        nvsmi_output_lines = f.readlines()
    # currently unused, but exists
    hardware_platform_line = nvsmi_output_lines[0]

    nvsmi_info = []
    for line in nvsmi_output_lines[1:]:
        nvsmi_dict = ast.literal_eval(line)
        num_gpus = nvsmi_dict['num_gpus']

        for i in range(num_gpus):
            if nvsmi_dict[i]['temp_celsius'] != 'N/A':
                temp_celsius_match = re.search(temp_celsius_pattern, nvsmi_dict[i]['temp_celsius'])
                curr_temp_celsius = int(temp_celsius_match.group(1))
                nvsmi_dict[i]['curr_temp_celsius'] = curr_temp_celsius

            if nvsmi_dict[i]['power_usage'] != 'N/A':
                power_usage_match = re.search(power_usage_pattern, nvsmi_dict[i]['power_usage'])
                curr_power_usage = int(power_usage_match.group(1))
                max_power_usage = int(power_usage_match.group(2))
                nvsmi_dict[i]['curr_power_usage'] = curr_power_usage
                nvsmi_dict[i]['max_power_usage'] = max_power_usage

            if nvsmi_dict[i]['memory_usage'] != 'N/A':
                memory_usage_match = re.search(memory_usage_pattern, nvsmi_dict[i]['memory_usage'])
                curr_memory_usage = int(memory_usage_match.group(1))
                max_memory_usage = int(memory_usage_match.group(2))
                nvsmi_dict[i]['curr_memory_usage'] = curr_memory_usage
                nvsmi_dict[i]['max_memory_usage'] = max_memory_usage

            if nvsmi_dict[i]['gpu_utilization'] != 'N/A':
                gpu_utilization_match = re.search(gpu_utilization_pattern, nvsmi_dict[i]['gpu_utilization'])
                gpu_utilization_percent = int(gpu_utilization_match.group(1))
                nvsmi_dict[i]['gpu_utilization_percent'] = gpu_utilization_percent

        nvsmi_info.append(nvsmi_dict) # TODO: make sure that times are strictly increasing in order

    return nvsmi_info


def write_nvsmi_output(nvsmi_output_path, nvsmi_output):
    print(f'write_nvsmi_output: {nvsmi_output_path}')

    # Extract the header of nvsmi output
    with open(nvsmi_output_path, 'r') as f:
        nvsmi_output_lines = f.readlines()
    hardware_platform_line = nvsmi_output_lines[0]

    new_nvsmi_output_lines = []
    with open(nvsmi_output_path, 'w') as f:
        f.write(hardware_platform_line)

        for nvsmi_dict in nvsmi_output:
            nvsmi_storage_dict = {}
            nvsmi_storage_dict['timestamp_readable'] = nvsmi_dict['timestamp_readable']
            nvsmi_storage_dict['timestamp_raw'] = nvsmi_dict['timestamp_raw']
            for i in range(nvsmi_dict['num_gpus']):
                gpu_storage_dict = {}
                gpu_dict = nvsmi_dict[i]

                gpu_storage_dict['temp_celsius'] = gpu_dict['temp_celsius']
                gpu_storage_dict['power_usage'] = gpu_dict['power_usage']
                gpu_storage_dict['memory_usage'] = gpu_dict['memory_usage']
                gpu_storage_dict['gpu_utilization'] = gpu_dict['gpu_utilization']
                nvsmi_storage_dict[i] = gpu_storage_dict

            nvsmi_storage_dict['num_gpus'] = nvsmi_dict['num_gpus']

            f.write(str(nvsmi_storage_dict) + '\n')
