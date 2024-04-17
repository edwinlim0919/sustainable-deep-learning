import argparse
import re
import ast
import matplotlib.pyplot as plt

from pathlib import Path


batch_input_tokens_pattern = r'batch_input_tokens\[(\d+)\]: \[(.*?)\]'
batch_output_tokens_pattern = r'batch_output_tokens\[(\d+)\]: \[(.*?)\]'

batch_input_lengths_pattern = r'batch_input_lengths\[(\d+)\]'
batch_output_lengths_pattern = r'batch_output_lengths\[(\d+)\]: \[(.*?)\]'
#batch_output_lengths_pattern = r'batch_output_lengths\[(\d+)\]'

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
            #batch_output_lengths = int(line.strip().split()[-1]) # TODO: Make experiment write just the int without the list wrapper
            length_list_str = batch_output_lengths_match.group(2)
            batch_output_lengths = int(ast.literal_eval(f'[{length_list_str}]')[0])
            bmark_info[curr_iteration]['batch_output_lengths'][batch_output_lengths_index] = batch_output_lengths

    return bmark_info


power_usage_pattern = r'(\d+)W / (\d+)W'
memory_usage_pattern = r'(\d+)MiB / (\d+)MiB'
gpu_utilization_pattern = r'(\d+)%'

def parse_nvsmi_output(nvsmi_output_path):
    print(f'parse_nvsmi_output: {nvsmi_output_path}')
    with open(nvsmi_output_path, 'r') as f:
        nvsmi_output_lines = f.readlines()
    hardware_platform_line = nvsmi_output_lines[0] # TODO: currently unused

    nvsmi_info = []
    for line in nvsmi_output_lines[1:]:
        nvsmi_dict = ast.literal_eval(line)

        power_usage_match = re.search(power_usage_pattern, nvsmi_dict['power_usage'])
        curr_power_usage = int(power_usage_match.group(1))
        max_power_usage = int(power_usage_match.group(2))
        nvsmi_dict['curr_power_usage'] = curr_power_usage
        nvsmi_dict['max_power_usage'] = max_power_usage

        memory_usage_match = re.search(memory_usage_pattern, nvsmi_dict['memory_usage'])
        curr_memory_usage = int(memory_usage_match.group(1))
        max_memory_usage = int(memory_usage_match.group(2))
        nvsmi_dict['curr_memory_usage'] = curr_memory_usage
        nvsmi_dict['max_memory_usage'] = max_memory_usage

        gpu_utilization_match = re.search(gpu_utilization_pattern, nvsmi_dict['gpu_utilization'])
        gpu_utilization_percent = int(gpu_utilization_match.group(1))
        nvsmi_dict['gpu_utilization_percent'] = gpu_utilization_percent

        nvsmi_info.append(nvsmi_dict) # TODO: make sure that times are strictly increasing in order

    return nvsmi_info


def plot_power_over_time(
    bmark_entries,
    plot_filename
):
    for bmark_entry in bmark_entries:
        model_size_GB = bmark_entry['model_size_GB']
        batch_size = bmark_entry['batch_size']
        max_sequence_length = bmark_entry['max_sequence_length']
        print(f'bmark_entry: {model_size_GB} {batch_size} {max_sequence_length}')

    testing = bmark_entries[0]
    for key, value in testing.items():
        print(f'testing: {key}')

    bmark_info = testing['bmark_info']
    # Extract timestamps from bmark_info
    curr_max_time = 0.0
    # each entry is (batch_start_time, batch_end_time)
    bmark_tuples = []
    for batch_iteration, batch_dict in bmark_info.items():
        #print(f'{batch_iteration}: {batch_dict}')
        batch_start_time = batch_dict['batch_start_time']
        batch_end_time = batch_dict['batch_end_time']

        # make sure timestamps are strictly increasing
        assert(batch_start_time > curr_max_time and
               batch_end_time > batch_start_time)
        curr_max_time = batch_end_time
        bmark_tuples.append((batch_start_time, batch_end_time))

    nvsmi_info = testing['nvsmi_info']
    # Extract timestamps and power usage from nvsmi_info
    #print(nvsmi_info[0])
    # each entry is (timestamp_raw, curr_power_usage, max_power_usage)
    #nvsmi_tuples = []
    timestamps = []
    curr_powers = []
    max_powers = []
    for nvsmi_dict in nvsmi_info:
        timestamp_raw = nvsmi_dict['timestamp_raw']
        curr_power_usage = nvsmi_dict['curr_power_usage']
        max_power_usage = nvsmi_dict['max_power_usage']

        timestamps.append(nvsmi_dict['timestamp_raw'])
        curr_powers.append(nvsmi_dict['curr_power_usage'])
        max_powers.append(nvsmi_dict['max_power_usage'])

    # make the timestamps start at 0
    timestamps_norm = [timestamp - timestamps[0] for timestamp in timestamps]

    plt.figure(figsize=(10, 5))
    plt.plot(timestamps_norm, curr_powers, label='Current GPU Power Usage')
    plt.axhline(y=max_powers[0], color='r', linestyle='--', label='Peak GPU Power Usage')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Power Usage (W)')
    plt.title('GPU Power Usage Over Time')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(plot_filename)


        #print(f'nvsmi_tuple: {timestamp_raw} {curr_power_usage} {max_power_usage}')
        #nvsmi_tuples.append((timestamp_raw, curr_power_usage, max_power_usage))

        #print(batch_start_time)
        #print(batch_end_time)

        #bmark_info = testing['bmark_info']
        #for key, value in bmark_info.items():
        #    print(f'bmark_info: {key}')


def main(args):
    bmark_output_paths = args.bmark_output_paths
    nvsmi_output_paths = args.nvsmi_output_paths
    bmark_params = args.bmark_params
    assert(len(bmark_output_paths) == len(nvsmi_output_paths) and 
           len(nvsmi_output_paths) == len(bmark_params))

    bmark_entries = []
    for i in range(len(bmark_output_paths)):
        bmark_entry = {}
        curr_bmark_params = bmark_params[i].split()
        model_size_GB = int(curr_bmark_params[0])
        batch_size = int(curr_bmark_params[1])
        max_sequence_length = int(curr_bmark_params[2])
        bmark_info = parse_bmark_output(bmark_output_paths[i])
        nvsmi_info = parse_nvsmi_output(nvsmi_output_paths[i])

        bmark_entry['model_size_GB'] = model_size_GB
        bmark_entry['batch_size'] = batch_size
        bmark_entry['max_sequence_length'] = max_sequence_length
        bmark_entry['bmark_info'] = bmark_info
        bmark_entry['nvsmi_info'] = nvsmi_info
        bmark_entries.append(bmark_entry)

    num_bmark_entries = len(bmark_entries)
    print(f'main num_bmark_entries: {num_bmark_entries}')

    if args.plot_power_over_time: # TODO: Only one plot can be generated at a time
        plot_power_over_time(bmark_entries, args.plot_filename)

    #print(bmark_entries[0])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--bmark_output_paths',
        type=str,
        nargs='+',
        required=True,
        help='paths to bmark output files'
    )
    parser.add_argument(
        '--nvsmi_output_paths',
        type=str,
        nargs='+',
        required=True,
        help='paths to nvsmi output files'
    )
    parser.add_argument(
        '--bmark_params',
        type=str,
        nargs='+',
        required=True,
        help='[model size] [batch size] [max sequence length]'
    )
    parser.add_argument(
        '--plot_power_over_time',
        default=False,
        action='store_true',
        help='specify this arg to plot power over time'
    )
    parser.add_argument(
        '--plot_filename',
        type=str,
        required=True,
        help='filename for specified plot'
    )
    args = parser.parse_args()
    main(args)
