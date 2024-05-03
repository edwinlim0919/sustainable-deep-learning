import argparse
import re
import ast
import matplotlib.pyplot as plt
from pathlib import Path

import transformer_model_scaling
import gpu_batch_exp_utils


def plot_throughput_vs_latency(
    bmark_entries,
    bmark_param_groups,
    plot_filename,
    plot_name
):
    # This is just for grouping different bmark data points into lines
    bmark_param_group_dicts = []
    for bmark_param_group in bmark_param_groups:
        group_split = bmark_param_group.split()
        bmark_param_group_dict = {}
        bmark_param_group_dict['model_size'] = int(group_split[0]) if group_split[0] != 'X' else 'X'
        bmark_param_group_dict['batch_size'] = int(group_split[1]) if group_split[1] != 'X' else 'X'
        bmark_param_group_dict['max_sequence_length'] = int(group_split[2]) if group_split[2] != 'X' else 'X'
        bmark_param_group_dict['gpu_type'] = group_split[3] if group_split[3] != 'X' else 'X'

        # For latency vs. throughput plots, track batch_sizes + avg tps + avg spt
        bmark_param_group_dict['batch_sizes'] = []
        bmark_param_group_dict['avg_tpss'] = []
        bmark_param_group_dict['avg_spts'] = []
        bmark_param_group_dicts.append(bmark_param_group_dict)

    for bmark_entry in bmark_entries:
        model_size = bmark_entry['model_size']
        batch_size = bmark_entry['batch_size']
        max_sequence_length = bmark_entry['max_sequence_length']
        gpu_type = bmark_entry['gpu_type']
        #batch_sweep_info = {
        #    'model_size': model_size,
        #    'batch_size': batch_size,
        #    'max_sequence_length': max_sequence_length,
        #    'gpu_type': gpu_type
        #}
        print(f'bmark_entry: {model_size} {batch_size} {max_sequence_length} {gpu_type}')

        bmark_info = bmark_entry['bmark_info']
        # tps = tokens per second
        batch_tps_sum = 0
        batch_spt_sum = 0
        num_iterations = 0
        for batch_iteration, batch_dict in bmark_info.items():
            batch_start_time = batch_dict['batch_start_time']
            batch_end_time = batch_dict['batch_end_time']
            batch_e2e_time = batch_end_time - batch_start_time

            batch_input_lengths_sum, batch_output_lengths_sum = 0, 0
            for batch_input_length_index, batch_input_lengths in batch_dict['batch_input_lengths'].items():
                batch_input_lengths_sum += batch_input_lengths
            for batch_output_length_index, batch_output_lengths in batch_dict['batch_output_lengths'].items():
                batch_output_lengths_sum += batch_output_lengths
            total_batch_generated_tokens = batch_output_lengths_sum - batch_input_lengths_sum

            # The average number of generated tokens for a single prompt in the batch
            avg_batch_generated_tokens = total_batch_generated_tokens / batch_size
            # The average seconds per token for a single prompt in the batch
            batch_spt = batch_e2e_time / avg_batch_generated_tokens
            batch_spt_sum += batch_spt

            # tps = tokens per second
            batch_tps = total_batch_generated_tokens / batch_e2e_time
            batch_tps_sum += batch_tps
            num_iterations += 1

        avg_tps = batch_tps_sum / num_iterations
        avg_spt = batch_spt_sum / num_iterations
        #bmark_entry['avg_tps'] = avg_tps
        #bmark_entry['avg_spt'] = avg_spt
        print(f'{model_size} {batch_size} {max_sequence_length} {gpu_type} {avg_tps} {avg_spt}')

        # group plotting points into the group_dicts
        bmark_param_match_found = False
        for bmark_param_group_dict in bmark_param_group_dicts:
            if (bmark_param_group_dict['model_size'] != 'X' and
                bmark_param_group_dict['model_size'] != model_size):
                continue
            if (bmark_param_group_dict['batch_size'] != 'X' and
                bmark_param_group_dict['batch_size'] != batch_size):
                continue
            if (bmark_param_group_dict['max_sequence_length'] != 'X' and
                bmark_param_group_dict['max_sequence_length'] != max_sequence_length):
                continue
            if (bmark_param_group_dict['gpu_type'] != 'X' and
                bmark_param_group_dict['gpu_type'] != gpu_type):
                continue

            # Only reach this point if a match is found
            bmark_param_match_found = True
            #bmark_param_group_dict['batch_sweep_info'].append(batch_sweep_info)
            bmark_param_group_dict['batch_sizes'].append(batch_size)
            bmark_param_group_dict['avg_tpss'].append(avg_tps)
            bmark_param_group_dict['avg_spts'].append(avg_spt)
            break

        # For each bmark_entry, should at least match to one of the plotting groups
        assert(bmark_param_match_found)

    plt.figure(figsize=(10, 5))
    for bmark_param_group_dict in bmark_param_group_dicts:
        for key, val in bmark_param_group_dict.items():
            print(f'{key}, {val}')

        model_size = bmark_param_group_dict['model_size']
        max_sequence_length = bmark_param_group_dict['max_sequence_length']
        gpu_type = bmark_param_group_dict['gpu_type']
        plt.plot(bmark_param_group_dict['avg_spts'], bmark_param_group_dict['avg_tpss'], label=f'{model_size} {gpu_type}', marker='o')
    plt.xlabel('Seconds Per Token')
    plt.ylabel('Tokens Per Second')
    plt.title(plot_name)
    plt.grid(True)
    plt.savefig(plot_filename)


    #plt.figure(figsize=(10, 5))
    #plt.plot(plot_batch_sizes, plot_gpu_utilization, marker='o', linestyle='-', color='blue')
    #plt.xlabel('Batch Size')
    #plt.ylabel('GPU Utilization (%)')
    #plt.title('V100 32GB PCIE Utilization w/ Llama 7B')
    #plt.xticks(plot_batch_sizes)
    #plt.grid(True)
    #plt.savefig('v10032gb_llama7b_utilization.png')



# Normalized token latency
# - The mean of every request's end-to-end latency divided by its output length
# - Input tokens also included in the output, but they also need preprocessing in the prefill stage, so included in calculation
#   TODO: Is this okay?
def plot_normalized_token_latency(
    bmark_entries,
    plot_filename,
    bmark_param_groups,
    excluded_tokens
):
    plt.figure(figsize=(10, 5))
    bmark_param_group_dicts = []
    for bmark_param_group in bmark_param_groups:
        group_split = bmark_param_group.split()
        bmark_param_group_dict = {}
        bmark_param_group_dict['model_size'] = int(group_split[0]) if group_split[0] != 'X' else 'X'
        bmark_param_group_dict['batch_size'] = int(group_split[1]) if group_split[1] != 'X' else 'X'
        bmark_param_group_dict['max_sequence_length'] = int(group_split[2]) if group_split[2] != 'X' else 'X'
        bmark_param_group_dict['gpu_type'] = group_split[3] if group_split[3] != 'X' else 'X'
        bmark_param_group_dict['batch_sweep_info'] = []
        bmark_param_group_dicts.append(bmark_param_group_dict)

    # holds dictionaries for holding just the plotting info
    #batch_sweep_infos = []

    for bmark_entry in bmark_entries:
        model_size = bmark_entry['model_size']
        batch_size = bmark_entry['batch_size']
        max_sequence_length = bmark_entry['max_sequence_length']
        gpu_type = bmark_entry['gpu_type']
        batch_sweep_info = {
            'model_size': model_size,
            'batch_size': batch_size,
            'max_sequence_length': max_sequence_length,
            'gpu_type': gpu_type
        }
        print(f'bmark_entry: {model_size} {batch_size} {max_sequence_length} {gpu_type}')

        # Extract timestamps from bmark_info
        bmark_info = bmark_entry['bmark_info']
        # keeping running sum of normalized token latencies to average at the end for this bmark
        normalized_token_latency_sum = 0
        included_normalized_token_latency_sum = 0
        e2e_batch_latency_sum = 0
        total_batch_output_lengths_sum = 0
        total_included_batch_output_lengths_sum = 0

        # each entry is (batch_start_time, batch_end_time)
        curr_max_time = 0.0
        num_iterations = 0
        for batch_iteration, batch_dict in bmark_info.items():
            num_iterations += 1
            batch_start_time = batch_dict['batch_start_time']
            batch_end_time = batch_dict['batch_end_time']

            # make sure timestamps are strictly increasing
            assert(batch_start_time > curr_max_time and
                   batch_end_time > batch_start_time)
            curr_max_time = batch_end_time
            # TODO: w/o continuous batching, latency of every request in batch is the same
            e2e_batch_latency = batch_end_time - batch_start_time
            e2e_batch_latency_sum += e2e_batch_latency

            batch_size = batch_dict['batch_size']
            total_batch_output_lengths = 0
            total_included_batch_output_lengths = 0
            for i in range(batch_size):
                batch_input_tokens = batch_dict['batch_input_tokens'][i]
                batch_input_lengths = batch_dict['batch_input_lengths'][i]
                batch_output_tokens = batch_dict['batch_output_tokens'][i]
                batch_output_lengths = batch_dict['batch_output_lengths'][i]

                # count non-padding tokens (or any excluded tokens)
                included_batch_input_lengths, included_batch_output_lengths = 0, 0
                excluded_batch_input_lengths, excluded_batch_output_lengths = 0, 0
                included_batch_input_tokens, included_batch_output_tokens = [], []
                excluded_batch_input_tokens, excluded_batch_output_tokens = [], []
                for token_id in batch_input_tokens: # TODO: these values currently are not used anywhere
                    assert(type(token_id) == int)
                    if token_id in excluded_tokens:
                        excluded_batch_input_tokens.append(token_id)
                        excluded_batch_input_lengths += 1
                    else:
                        included_batch_input_tokens.append(token_id)
                        included_batch_input_lengths += 1
                for token_id in batch_output_tokens:
                    assert(type(token_id) == int)
                    if token_id in excluded_tokens:
                        excluded_batch_output_tokens.append(token_id)
                        excluded_batch_output_lengths += 1
                    else:
                        included_batch_output_tokens.append(token_id)
                        included_batch_output_lengths += 1

                # add to token sums for this batch
                total_batch_output_lengths += batch_output_lengths
                total_included_batch_output_lengths += included_batch_output_lengths

                # verify lengths for this batch
                assert(len(batch_input_tokens) == batch_input_lengths and
                       len(batch_output_tokens) == (included_batch_output_lengths + excluded_batch_output_lengths))

            # calculate normalized token latencies for this current batch
            batch_normalized_token_latency = e2e_batch_latency / total_batch_output_lengths
            included_batch_normalized_token_latency = e2e_batch_latency / total_included_batch_output_lengths
            total_batch_output_lengths_sum += total_batch_output_lengths
            total_included_batch_output_lengths_sum += total_included_batch_output_lengths

            # add latencies to running sums
            normalized_token_latency_sum += batch_normalized_token_latency
            included_normalized_token_latency_sum += included_batch_normalized_token_latency

        # calculate actual normalized token latencies for entire bmark
        batch_sweep_info['normalized_token_latency'] = normalized_token_latency_sum / num_iterations
        batch_sweep_info['included_normalized_token_latency'] = included_normalized_token_latency_sum / num_iterations
        batch_sweep_info['avg_e2e_batch_latency'] = e2e_batch_latency_sum / num_iterations
        batch_sweep_info['avg_output_tokens_per_batch'] = total_batch_output_lengths_sum / num_iterations
        batch_sweep_info['avg_included_output_tokens_per_batch'] = total_included_batch_output_lengths_sum / num_iterations
        #batch_sweep_infos.append(batch_sweep_info)

        # calculate avg FLOPs per batch 7B
        avg_flops_per_batch = transformer_model_scaling.calculate_transformer_flops(
            4096,
            32,
            4096,
            4096,
            int(batch_sweep_info['avg_output_tokens_per_batch'])
        )
        batch_sweep_info['avg_flops_per_batch'] = avg_flops_per_batch
        batch_sweep_info['avg_tflops_per_batch'] = avg_flops_per_batch / (10 ** 12)

        # calculate TFLOPs achievable in avg batch time by a100 (624 FP16 TFLOPS)
        #batch_sweep_info['peak_TFLOPs_in_batch_time'] = batch_sweep_info['avg_e2e_batch_latency'] * 624
        #batch_sweep_info['avg_GPU_utilization'] = batch_sweep_info['avg_tflops_per_batch'] / batch_sweep_info['peak_TFLOPs_in_batch_time']

        # TFLOPs achievable v100 (130 TFLOPS)
        batch_sweep_info['peak_TFLOPs_in_batch_time'] = batch_sweep_info['avg_e2e_batch_latency'] * 130
        batch_sweep_info['avg_GPU_utilization'] = batch_sweep_info['avg_tflops_per_batch'] / batch_sweep_info['peak_TFLOPs_in_batch_time']

        # group things in to their bmark param group
        for bmark_param_group_dict in bmark_param_group_dicts:
            bmark_param_match_found = False
            if (bmark_param_group_dict['model_size'] != 'X' and
                bmark_param_group_dict['model_size'] != model_size):
                continue
            if (bmark_param_group_dict['batch_size'] != 'X' and
                bmark_param_group_dict['batch_size'] != batch_size):
                continue
            if (bmark_param_group_dict['max_sequence_length'] != 'X' and
                bmark_param_group_dict['max_sequence_length'] != max_sequence_length):
                continue
            if (bmark_param_group_dict['gpu_type'] != 'X' and
                bmark_param_group_dict['gpu_type'] != gpu_type):
                continue

            # Only reach this point if a match is found
            bmark_param_match_found = True
            bmark_param_group_dict['batch_sweep_info'].append(batch_sweep_info)
            break

        # For each bmark_entry, should at least match to one of the plotting groups
        assert(bmark_param_match_found)

    plot_batch_sizes = []
    plot_gpu_utilization = []
    for bmark_param_group_dict in bmark_param_group_dicts:
        #for key, value in bmark_param_group_dict.items():
        #    print(f'{key}: {value}')
        batch_sweep_info = bmark_param_group_dict['batch_sweep_info']
        for info_dict in batch_sweep_info:
            print(f'{info_dict["batch_size"]}: {info_dict["avg_GPU_utilization"]}')
            plot_batch_sizes.append(info_dict['batch_size'])
            plot_gpu_utilization.append(info_dict['avg_GPU_utilization'] * 100)

    plt.figure(figsize=(10, 5))
    #plt.bar(plot_batch_sizes, plot_gpu_utilization, color='blue')
    plt.plot(plot_batch_sizes, plot_gpu_utilization, marker='o', linestyle='-', color='blue')

    plt.xlabel('Batch Size')
    plt.ylabel('GPU Utilization (%)')
    #plt.title('A100 40GB SXM Utilization w/ Llama 7B')
    plt.title('V100 32GB PCIE Utilization w/ Llama 7B')
    plt.xticks(plot_batch_sizes)
    plt.grid(True)
    plt.savefig('v10032gb_llama7b_utilization.png')

    # sanity prints
    #for batch_sweep_info in batch_sweep_infos:
    #    for key, value in batch_sweep_info.items():
    #        print(f'{key}: {value}')
    #    print()

    #example_bmark_info = bmark_entries[0]['bmark_info']
    #for batch_iteration, batch_dict in example_bmark_info.items():
    #    example_batch_dict = batch_dict
    #for key, value in batch_dict.items():
    #    print(f'{key}: {value}')


def plot_power_over_time(
    bmark_entries,
    plot_filename,
    gpu_idx
    #plot_sequence_lengths,
    #plot_batch_sizes
):
    plt.figure(figsize=(10, 5))
    min_bmark_nvsmi_time_start_diff = float('inf')
    timestamps_list = []
    curr_powers_list = []
    max_powers_list = []
    bmark_tuples_list = []
    bmark_entry_list = []

    for bmark_entry in bmark_entries:
        model_size = bmark_entry['model_size']
        batch_size = bmark_entry['batch_size']
        max_sequence_length = bmark_entry['max_sequence_length']
        gpu_type = bmark_entry['gpu_type']
        print(f'bmark_entry: {model_size} {batch_size} {max_sequence_length} {gpu_type}')

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
            #for key, val in nvsmi_dict.items():
            #    print(f'{key}: {val}')
            #    print(f'{key}: {type(key)}')
            #print(f'{gpu_idx}: {type(gpu_idx)}')
            gpu_idx_dict = nvsmi_dict[gpu_idx]
            curr_power_usage = gpu_idx_dict['curr_power_usage']
            max_power_usage = gpu_idx_dict['max_power_usage']

            timestamps.append(nvsmi_dict['timestamp_raw'])
            curr_powers.append(gpu_idx_dict['curr_power_usage'])
            max_powers.append(gpu_idx_dict['max_power_usage'])

        # Make the timestamps start at the same place
        bmark_nvsmi_time_start_diff = bmark_tuples[0][0] - timestamps[0]
        if bmark_nvsmi_time_start_diff < min_bmark_nvsmi_time_start_diff:
            min_bmark_nvsmi_time_start_diff = bmark_nvsmi_time_start_diff

        timestamps_list.append(timestamps)#_norm)
        curr_powers_list.append(curr_powers)
        max_powers_list.append(max_powers)
        bmark_tuples_list.append(bmark_tuples)
        bmark_entry_list.append(bmark_entry)

    #for tl in timestamps_list:
    #    print(f'tl: {tl}')

    # making all the plots start execution at the same time point
    assert(len(timestamps_list) == len(curr_powers_list) and
           len(curr_powers_list) == len(max_powers_list) and
           len(max_powers_list) == len(bmark_tuples_list))

    # TODO: oh my god...
    for j in range(len(curr_powers_list)):
        lst = curr_powers_list[j]
    #for lst in curr_powers_list:
        print(f'lst: {lst}')

        whatthefuck = []
        for i in range(len(lst)):
            whatthefuck.append(i)

        assert(len(whatthefuck) == len(lst))
            #print(i)
        #curr_fake_timestamps_list = []
        #for i in range(len(lst)):
        #    curr_fake_timestamps_list.append(i)

    #for i in range(len(timestamps_list)):
        print(j)
        bmark_entry = bmark_entry_list[j]
        model_size = bmark_entry['model_size']
        batch_size = bmark_entry['batch_size']
        max_sequence_length = bmark_entry['max_sequence_length']
        gpu_type = bmark_entry['gpu_type']

    #    timestamps = timestamps_list[i]
    #    curr_powers = curr_powers_list[i]
    #    max_powers = max_powers_list[i]
    #    bmark_tuples = bmark_tuples_list[i]

    #    bmark_nvsmi_time_start_diff = bmark_tuples[0][0] - timestamps[0]
    #    diff_from_min = bmark_nvsmi_time_start_diff - min_bmark_nvsmi_time_start_diff

    #    timestamps_norm = [timestamp - timestamps[0] for timestamp in timestamps]
    #    timestamps_adjusted = [timestamp - diff_from_min for timestamp in timestamps_norm]

    #    assert(len(timestamps_adjusted) == len(curr_powers) and
    #           len(curr_powers) == len(max_powers))
    #    plot_timestamps = []
    #    plot_curr_powers = []

    #    for i in range(len(timestamps_adjusted)):
    #        if timestamps_adjusted[i] >= 0:
    #            plot_timestamps.append(timestamps_adjusted[i])
    #            plot_curr_powers.append(curr_powers[i])

    #    # TODO: oh my god...
    #    #makeshift_plot_timestamps = 

    #    if j == 3 or j == 4:

    #        plt.plot(whatthefuck, lst, label=f'maxseq: {max_sequence_length}, gpu: {gpu_type}, wq4')
    #    else:
    #        plt.plot(whatthefuck, lst, label=f'maxseq: {max_sequence_length}, gpu: {gpu_type}')

    #plt.axhline(y=250, color='r', linestyle='--', label='Peak v100 Power Usage')
    #plt.axhline(y=400, color='r', linestyle='--', label='Peak a100 Power Usage')
    #plt.xlabel('Time (seconds)')
    #plt.ylabel('Power Usage (W)')
    #plt.title(f'GPU Power Usage Llama{model_size}B')
    #plt.legend()
    #plt.grid(True)
    #plt.tight_layout()
    #plt.savefig(plot_filename)


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
        bmark_param_group_dict['model_size'] = int(group_split[0]) if group_split[0] != 'X' else 'X'
        bmark_param_group_dict['batch_size'] = int(group_split[1]) if group_split[1] != 'X' else 'X'
        bmark_param_group_dict['max_sequence_length'] = int(group_split[2]) if group_split[2] != 'X' else 'X'
        bmark_param_group_dict['avg_batch_latencies'] = []
        bmark_param_group_dicts.append(bmark_param_group_dict)

    for bmark_entry in bmark_entries:
        model_size = bmark_entry['model_size']
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
        print(f'PLOT_AVERAGE_BATCH_LATENCY model_size: {model_size}, batch_size: {batch_size}, max_sequence_length: {max_sequence_length}, avg_batch_latency: {avg_batch_latency}, batch_size: {batch_size}')
        avg_batch_latencies_dict = {
            'batch_size': batch_size,
            'avg_batch_latency': avg_batch_latency
        }

        # Identify which bmark_param_group_dict to append to
        for bmark_param_group_dict in bmark_param_group_dicts:
            bmark_param_match_found = False
            if (bmark_param_group_dict['model_size'] != 'X' and
                bmark_param_group_dict['model_size'] != model_size):
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

        plt.plot(batch_sizes, avg_latencies, label=f'Llama {bmark_param_group_dict["model_size"]}B Max {bmark_param_group_dict["max_sequence_length"]} Sequence Length)')

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
        model_size = int(curr_bmark_params[0])
        batch_size = int(curr_bmark_params[1])
        max_sequence_length = int(curr_bmark_params[2])
        gpu_type = curr_bmark_params[3]
        bmark_info = gpu_batch_exp_utils.parse_bmark_output(bmark_output_paths[i])
        nvsmi_info = gpu_batch_exp_utils.parse_nvsmi_output(nvsmi_output_paths[i])

        bmark_entry['model_size'] = model_size
        bmark_entry['batch_size'] = batch_size
        bmark_entry['max_sequence_length'] = max_sequence_length
        bmark_entry['gpu_type'] = gpu_type
        bmark_entry['bmark_info'] = bmark_info
        bmark_entry['nvsmi_info'] = nvsmi_info
        bmark_entries.append(bmark_entry)

    if args.plot_power_over_time: # TODO: Only one plot can be generated at a time
        #if not args.plot_sequence_lengths:
        #    raise ValueError('supply plot_sequence_lengths argument for plot_power_over_time')
        #if not args.plot_batch_sizes:
        #    raise ValueError('supply plot_batch_sizes argument for plot_power_over_time')
        plot_power_over_time(
            bmark_entries,
            args.plot_filename,
            args.gpu_idx
            #args.plot_sequence_lengths,
            #args.plot_batch_sizes
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
        #if not args.plot_sequence_lengths:
        #    raise ValueError('supply plot_sequence_lengths argument for plot_average_batch_latency')
        #if not args.plot_batch_sizes:
        #    raise ValueError('supply plot_batch_sizes argument for plot_average_batch_latency')
        plot_normalized_token_latency(
            bmark_entries,
            args.plot_filename,
            args.bmark_param_groups,
            args.excluded_tokens
        )
    if args.plot_throughput_vs_latency:
        plot_throughput_vs_latency(
            bmark_entries,
            args.bmark_param_groups,
            args.plot_filename,
            args.plot_name
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
        help='[model size] [batch size] [max sequence length] [gpu type]'
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
        '--plot_throughput_vs_latency',
        default=False,
        action='store_true',
        help='specify this arg to plot throughput vs latency'
    )
    parser.add_argument(
        '--plot_filename',
        type=str,
        required=True,
        help='filename for specified plot'
    )
    parser.add_argument(
        '--plot_name',
        type=str,
        required=True,
        help='title for specified plot'
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
    parser.add_argument(
        '--excluded_tokens',
        type=int,
        nargs='+',
        help='specify tokens ids such as padding token ids to exclude from useful work done'
    )
    parser.add_argument(
        '--gpu_idx',
        type=int,
        help='specify which idx of GPU for nvsmi info'
    )
    args = parser.parse_args()
    main(args)
