import argparse
import re
import ast
import matplotlib.pyplot as plt

from pathlib import Path


batch_input_tokens_pattern = r'batch_input_tokens\[(\d+)\]: \[(.*?)\]'
batch_output_tokens_pattern = r'batch_output_tokens\[(\d+)\]: \[(.*?)\]'

batch_input_lengths_pattern = r'batch_input_lengths\[(\d+)\]'
#batch_output_lengths_pattern = r'batch_output_lengths\[(\d+)\]: \[(.*?)\]'
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

            #batch_output_lengths_match = re.search(batch_output_lengths_pattern, line)
            #batch_output_lengths_index = int(batch_output_lengths_match.group(1))
            #length_list_str = batch_output_lengths_match.group(2)
            #batch_output_lengths = int(ast.literal_eval(f'[{length_list_str}]')[0])
            #bmark_info[curr_iteration]['batch_output_lengths'][batch_output_lengths_index] = batch_output_lengths

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


# Normalized token latency
# - The mean of every request's end-to-end latency divided by its output length
def plot_normalized_token_latency(
    bmark_entries,
    plot_filename,
    plot_sequence_lengths,
    plot_batch_sizes
):
    for bmark_entry in bmark_entries:
        model_size_GB = bmark_entry['model_size_GB']
        batch_size = bmark_entry['batch_size']
        max_sequence_length = bmark_entry['max_sequence_length']
        if max_sequence_length not in plot_sequence_lengths:
            continue
        if batch_size not in plot_batch_sizes:
            continue
        print(f'bmark_entry: {model_size_GB} {batch_size} {max_sequence_length}')

        # Extract timestamps from bmark_info
        bmark_info = bmark_entry['bmark_info']
        # each entry is (batch_start_time, batch_end_time)
        curr_max_time = 0.0
        for batch_iteration, batch_dict in bmark_info.items():
            batch_start_time = batch_dict['batch_start_time']
            batch_end_time = batch_dict['batch_end_time']

            # make sure timestamps are strictly increasing
            assert(batch_start_time > curr_max_time and
                   batch_end_time > batch_start_time)
            curr_max_time = batch_end_time

            # TODO: w/o continuous batching, latency of every request in batch is the same
            batch_e2e_latency = batch_end_time - batch_start_time

    example_bmark_info = bmark_entries[0]['bmark_info']
    for batch_iteration, batch_dict in example_bmark_info.items():
        example_batch_dict = batch_dict
    for key, value in batch_dict.items():
        print(f'{key}: {value}')

        #print(f'batch_start_time: {batch_start_time}, batch_end_time: {batch_end_time}')

    #print(f'bmark_entries[0]: {bmark_entries[0]}')


def plot_power_over_time(
    bmark_entries,
    plot_filename,
    plot_sequence_lengths,
    plot_batch_sizes
):
    plt.figure(figsize=(10, 5))
    min_bmark_nvsmi_time_start_diff = float('inf')
    timestamps_list = []
    curr_powers_list = []
    max_powers_list = []
    bmark_tuples_list = []
    bmark_entry_list = []

    for bmark_entry in bmark_entries:
        model_size_GB = bmark_entry['model_size_GB']
        batch_size = bmark_entry['batch_size']
        max_sequence_length = bmark_entry['max_sequence_length']
        if max_sequence_length not in plot_sequence_lengths:
            continue
        if batch_size not in plot_batch_sizes:
            continue
        print(f'bmark_entry: {model_size_GB} {batch_size} {max_sequence_length}')

        # Extract timestamps from bmark_info
        bmark_info = bmark_entry['bmark_info']
        # each entry is (batch_start_time, batch_end_time)
        bmark_tuples = []
        curr_max_time = 0.0
        for batch_iteration, batch_dict in bmark_info.items():
            batch_start_time = batch_dict['batch_start_time']
            batch_end_time = batch_dict['batch_end_time']

            # make sure timestamps are strictly increasing
            assert(batch_start_time > curr_max_time and
                   batch_end_time > batch_start_time)
            curr_max_time = batch_end_time
            bmark_tuples.append((batch_start_time, batch_end_time))

        # Extract timestamps and power usage from nvsmi_info
        nvsmi_info = bmark_entry['nvsmi_info']
        # each entry is (timestamp_raw, curr_power_usage, max_power_usage)
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

        # Make the timestamps start at the same place
        bmark_nvsmi_time_start_diff = bmark_tuples[0][0] - timestamps[0]
        if bmark_nvsmi_time_start_diff < min_bmark_nvsmi_time_start_diff:
            min_bmark_nvsmi_time_start_diff = bmark_nvsmi_time_start_diff

        timestamps_list.append(timestamps)#_norm)
        curr_powers_list.append(curr_powers)
        max_powers_list.append(max_powers)
        bmark_tuples_list.append(bmark_tuples)
        bmark_entry_list.append(bmark_entry)

    # making all the plots start execution at the same time point
    assert(len(timestamps_list) == len(curr_powers_list) and
           len(curr_powers_list) == len(max_powers_list) and
           len(max_powers_list) == len(bmark_tuples_list))
    for i in range(len(timestamps_list)):
        bmark_entry = bmark_entry_list[i]
        batch_size = bmark_entry['batch_size']

        timestamps = timestamps_list[i]
        curr_powers = curr_powers_list[i]
        max_powers = max_powers_list[i]
        bmark_tuples = bmark_tuples_list[i]

        bmark_nvsmi_time_start_diff = bmark_tuples[0][0] - timestamps[0]
        diff_from_min = bmark_nvsmi_time_start_diff - min_bmark_nvsmi_time_start_diff

        timestamps_norm = [timestamp - timestamps[0] for timestamp in timestamps]
        timestamps_adjusted = [timestamp - diff_from_min for timestamp in timestamps_norm]

        assert(len(timestamps_adjusted) == len(curr_powers) and
               len(curr_powers) == len(max_powers))
        plot_timestamps = []
        plot_curr_powers = []

        for i in range(len(timestamps_adjusted)):
            if timestamps_adjusted[i] >= 0:
                plot_timestamps.append(timestamps_adjusted[i])
                plot_curr_powers.append(curr_powers[i])
        plt.plot(plot_timestamps, plot_curr_powers, label=f'Measured GPU Power Usage (batch size {batch_size})')

    plt.axhline(y=max_powers[0], color='r', linestyle='--', label='Peak GPU Power Usage')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Power Usage (W)')
    plt.title(f'GPU Power Usage Llama{model_size_GB}B Max Seq. Len(s) {plot_sequence_lengths}')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(plot_filename)


def plot_average_batch_latency(
    bmark_entries,
    plot_filename,
    plot_sequence_lengths,
    plot_batch_sizes,
    bmark_param_groups
):
    plt.figure(figsize=(10, 5))

    bmark_param_group_dicts = []
    for bmark_param_group in bmark_param_groups:
        group_split = bmark_param_group.split()
        bmark_param_group_dict = {}
        bmark_param_group_dict['model_size_GB'] = int(group_split[0]) if group_split[0] != 'X' else 'X'
        bmark_param_group_dict['batch_size'] = int(group_split[1]) if group_split[1] != 'X' else 'X'
        bmark_param_group_dict['max_sequence_length'] = int(group_split[2]) if group_split[2] != 'X' else 'X'
        bmark_param_group_dict['avg_batch_latencies'] = []
        bmark_param_group_dicts.append(bmark_param_group_dict)

    for bmark_entry in bmark_entries:
        model_size_GB = bmark_entry['model_size_GB']
        batch_size = bmark_entry['batch_size']
        max_sequence_length = bmark_entry['max_sequence_length']
        if max_sequence_length not in plot_sequence_lengths:
            continue
        if batch_size not in plot_batch_sizes:
            continue

        # Extract timestamps from bmark_info
        bmark_info = bmark_entry['bmark_info']
        # each entry is (batch_start_time, batch_end_time)
        bmark_tuples = []
        curr_max_time = 0.0
        batch_latency_sum = 0.0
        num_iterations = len(bmark_info)

        for batch_iteration, batch_dict in bmark_info.items():
            batch_start_time = batch_dict['batch_start_time']
            batch_end_time = batch_dict['batch_end_time']

            # make sure timestamps are strictly increasing
            assert(batch_start_time > curr_max_time and
                   batch_end_time > batch_start_time)
            curr_max_time = batch_end_time
            batch_latency = batch_end_time - batch_start_time
            batch_latency_sum += batch_latency

        avg_batch_latency = batch_latency_sum / num_iterations
        print(f'PLOT_AVERAGE_BATCH_LATENCY model_size_GB: {model_size_GB}, batch_size: {batch_size}, max_sequence_length: {max_sequence_length}, avg_batch_latency: {avg_batch_latency}, batch_size: {batch_size}')
        avg_batch_latencies_dict = {
            'batch_size': batch_size,
            'avg_batch_latency': avg_batch_latency
        }

        # Identify which bmark_param_group_dict to append to
        for bmark_param_group_dict in bmark_param_group_dicts:
            bmark_param_match_found = False
            if (bmark_param_group_dict['model_size_GB'] != 'X' and
                bmark_param_group_dict['model_size_GB'] != model_size_GB):
                continue
            if (bmark_param_group_dict['batch_size'] != 'X' and
                bmark_param_group_dict['batch_size'] != batch_size):
                continue
            if (bmark_param_group_dict['max_sequence_length'] != 'X' and
                bmark_param_group_dict['max_sequence_length'] != max_sequence_length):
                continue

            # Only reach this point if a match is found
            bmark_param_match_found = True
            bmark_param_group_dict['avg_batch_latencies'].append(avg_batch_latencies_dict)
            break

        # For each bmark_entry, should at least match to one of the plotting groups
        assert(bmark_param_match_found)

    for bmark_param_group_dict in bmark_param_group_dicts:
        batch_sizes = []
        avg_latencies = []
        for avg_batch_latencies_dict in bmark_param_group_dict['avg_batch_latencies']:
            batch_sizes.append(avg_batch_latencies_dict['batch_size'])
            avg_latencies.append(avg_batch_latencies_dict['avg_batch_latency'])

        plt.plot(batch_sizes, avg_latencies, label=f'Llama {bmark_param_group_dict["model_size_GB"]}B Max {bmark_param_group_dict["max_sequence_length"]} Sequence Length)')

    # Minimum batch size of 1
    plt.xlim(left=1)
    plt.xlabel('batch size')
    plt.ylabel('Time (seconds)')
    plt.title(f'Avg. Batch Latency vs. Batch Size')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(plot_filename)



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

    if args.plot_power_over_time: # TODO: Only one plot can be generated at a time
        if not args.plot_sequence_lengths:
            raise ValueError('supply plot_sequence_lengths argument for plot_power_over_time')
        if not args.plot_batch_sizes:
            raise ValueError('supply plot_batch_sizes argument for plot_power_over_time')
        plot_power_over_time(
            bmark_entries,
            args.plot_filename,
            args.plot_sequence_lengths,
            args.plot_batch_sizes
        )
    if args.plot_average_batch_latency:
        if not args.plot_sequence_lengths:
            raise ValueError('supply plot_sequence_lengths argument for plot_average_batch_latency')
        if not args.plot_batch_sizes:
            raise ValueError('supply plot_batch_sizes argument for plot_average_batch_latency')
        if not args.bmark_param_groups:
            raise ValueError('supply bmark_param_groups for plot_average_batch_latency')
        plot_average_batch_latency(
            bmark_entries,
            args.plot_filename,
            args.plot_sequence_lengths,
            args.plot_batch_sizes,
            args.bmark_param_groups
        )
    if args.plot_normalized_token_latency:
        if not args.plot_sequence_lengths:
            raise ValueError('supply plot_sequence_lengths argument for plot_average_batch_latency')
        if not args.plot_batch_sizes:
            raise ValueError('supply plot_batch_sizes argument for plot_average_batch_latency')
        plot_normalized_token_latency(
            bmark_entries,
            args.plot_filename,
            args.plot_sequence_lengths,
            args.plot_batch_sizes
        )


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
        '--bmark_param_groups',
        type=str,
        nargs='+',
        help='[model size] [batch size] [max sequence length] (specify "X" for any value)'
    )
    parser.add_argument(
        '--plot_power_over_time',
        default=False,
        action='store_true',
        help='specify this arg to plot power over time'
    )
    parser.add_argument(
        '--plot_average_batch_latency',
        default=False,
        action='store_true',
        help='specify this arg to plot average batch latency'
    )
    parser.add_argument(
        '--plot_normalized_token_latency',
        default=False,
        action='store_true',
        help='specify this arg to plot normalized token latency'
    )
    parser.add_argument(
        '--plot_filename',
        type=str,
        required=True,
        help='filename for specified plot'
    )
    parser.add_argument(
        '--plot_sequence_lengths',
        type=int,
        nargs='+',
        help='specify which sequence length to generate the plot for'
    )
    parser.add_argument(
        '--plot_batch_sizes',
        type=int,
        nargs='+',
        help='specify which batch sizes to generate the plot for'
    )
    args = parser.parse_args()
    main(args)
