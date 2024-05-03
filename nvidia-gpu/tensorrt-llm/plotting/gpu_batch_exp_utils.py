import argparse
import re
import ast
import matplotlib.pyplot as plt
from pathlib import Path


batch_input_tokens_pattern = r'batch_input_tokens\[(\d+)\]: \[(.*?)\]'
batch_output_tokens_pattern = r'batch_output_tokens\[(\d+)\]: \[(.*?)\]'

batch_input_lengths_pattern = r'batch_input_lengths\[(\d+)\]'
batch_output_lengths_pattern = r'batch_output_lengths\[(\d+)\]'

def parse_bmark_output(bmark_output_path):
    print(f'parse_bmark_output: {bmark_output_path}')
    with open(bmark_output_path, 'r') as f:
        bmark_output_lines = f.readlines()

    engine_path_line = bmark_output_lines[0]
    tokenizer_path_line = bmark_output_lines[1]
    num_iterations_line = bmark_output_lines[2]
    num_iterations = int(num_iterations_line.split()[-1])
    bmark_info = {}

    for line in bmark_output_lines[3:]:
        if 'iteration' in line:
            curr_iteration = int(line.strip().split()[-1])
            bmark_info[curr_iteration] = {}
            bmark_info[curr_iteration]['batch_input_tokens'] = {}
            bmark_info[curr_iteration]['batch_input_lengths'] = {}
            bmark_info[curr_iteration]['batch_output_tokens'] = {}
            bmark_info[curr_iteration]['batch_output_lengths'] = {}

        if 'batch_size' in line:
            bmark_info[curr_iteration]['batch_size'] = int(line.strip().split()[-1])

        if 'max_input_tokens' in line:
            bmark_info[curr_iteration]['max_input_tokens'] = int(line.strip().split()[-1])

        if 'max_output_tokens' in line:
            bmark_info[curr_iteration]['max_output_tokens'] = int(line.strip().split()[-1])

        if 'batch_start_time' in line:
            bmark_info[curr_iteration]['batch_start_time'] = float(line.strip().split()[-1])

        if 'batch_end_time' in line:
            bmark_info[curr_iteration]['batch_end_time'] = float(line.strip().split()[-1])

        if 'batch_input_tokens' in line:
            batch_input_tokens_match = re.search(batch_input_tokens_pattern, line)
            batch_input_tokens_index = int(batch_input_tokens_match.group(1))
            token_list_str = batch_input_tokens_match.group(2)
            batch_input_tokens_list = ast.literal_eval(f'[{token_list_str}]')
            bmark_info[curr_iteration]['batch_input_tokens'][batch_input_tokens_index] = batch_input_tokens_list

        if 'batch_input_lengths' in line:
            batch_input_lengths_match = re.search(batch_input_lengths_pattern, line)
            batch_input_lengths_index = int(batch_input_lengths_match.group(1))
            batch_input_lengths = int(line.strip().split()[-1])
            bmark_info[curr_iteration]['batch_input_lengths'][batch_input_lengths_index] = batch_input_lengths

        if 'batch_output_tokens' in line:
            batch_output_tokens_match = re.search(batch_output_tokens_pattern, line)
            batch_output_tokens_index = int(batch_output_tokens_match.group(1))
            token_list_str = batch_output_tokens_match.group(2)
            batch_output_tokens_list = ast.literal_eval(f'[{token_list_str}]')
            bmark_info[curr_iteration]['batch_output_tokens'][batch_output_tokens_index] = batch_output_tokens_list

        if 'batch_output_lengths' in line:
            batch_output_lengths_match = re.search(batch_output_lengths_pattern, line)
            batch_output_lengths_index = int(batch_output_lengths_match.group(1))
            batch_output_lengths = int(line.strip().split()[-1])
            bmark_info[curr_iteration]['batch_output_lengths'][batch_output_lengths_index] = batch_output_lengths

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

                #nvsmi_storage_dict[i] = nvsmi_dict[i]
                nvsmi_storage_dict[i] = gpu_storage_dict
            nvsmi_storage_dict['num_gpus'] = nvsmi_dict['num_gpus']

            f.write(str(nvsmi_storage_dict) + '\n')
