import argparse
import re
import ast
import matplotlib.pyplot as plt
import math
from pathlib import Path

import numpy as np
import gpu_batch_exp_utils


# Group experiment data based on the parameter groups passed
def group_experiment_data(
    bmark_entries,
    bmark_param_groups,
    plotting_metrics
):
    bmark_param_group_dicts = []
    for bmark_param_group in bmark_param_groups:
        group_split = bmark_param_group.split()
        bmark_param_group_dict = {}
        bmark_param_group_dict['model_size'] = group_split[0] if group_split[0] != 'X' else 'X'
        bmark_param_group_dict['batch_size'] = int(group_split[1]) if group_split[1] != 'X' else 'X'
        bmark_param_group_dict['max_sequence_length'] = int(group_split[2]) if group_split[2] != 'X' else 'X'
        bmark_param_group_dict['gpu_type'] = group_split[3] if group_split[3] != 'X' else 'X'

        for plotting_metric in plotting_metrics:
            bmark_param_group_dict[plotting_metric] = []
        bmark_param_group_dicts.append(bmark_param_group_dict)

    return bmark_param_group_dicts


# Calcaulates TBT (Time Between Tokens) for different bmark data points and adds information to plotting dicts
def calculate_avg_tbt(
    bmark_entries,
    bmark_param_groups,
    excluded_tokens,
    plotting_knob,
    bmark_param_group_dicts
):
    for bmark_entry in bmark_entries:
        model_size = bmark_entry['model_size']
        batch_size = bmark_entry['batch_size']
        max_sequence_length = bmark_entry['max_sequence_length']
        gpu_type = bmark_entry['gpu_type']
        bmark_info = bmark_entry['bmark_info']

        # For this given bmark data point, keep track of running tbt sum to calculate avg at the end
        tbt_sum = 0
        # Manually track the number of parsed iterations for edge cases
        num_iterations = 0

        for batch_iteration, batch_dict in bmark_info.items():
            batch_start_time = batch_dict['batch_start_time']
            batch_end_time = batch_dict['batch_end_time']
            batch_e2e_time = batch_end_time - batch_start_time

            batch_input_lengths_sum, batch_output_lengths_sum = 0, 0
            batch_input_tokens_items = batch_dict['batch_input_tokens'].items()
            batch_output_tokens_items = batch_dict['batch_output_tokens'].items()
            assert(len(batch_input_tokens_items) == len(batch_output_tokens_items))

            for (batch_input_tokens_index, batch_input_tokens), (batch_output_tokens_index, batch_output_tokens) in zip(batch_input_tokens_items, batch_output_tokens_items):
                batch_input_length, batch_output_length = 0, 0
                for token in batch_input_tokens:
                    if token not in excluded_tokens:
                        batch_input_length += 1
                for token in batch_output_tokens:
                    if token not in excluded_tokens:
                        batch_output_length += 1
                batch_input_lengths_sum += batch_input_length
                batch_output_lengths_sum += batch_output_length

            assert(batch_size == len(batch_input_tokens_items))
            total_batch_generated_tokens = batch_output_lengths_sum - batch_input_lengths_sum
            # if no tokens generated, skip this iteration
            if total_batch_generated_tokens == 0:
                continue

            avg_batch_generated_tokens = total_batch_generated_tokens / batch_size
            batch_tbt_avg = batch_e2e_time / avg_batch_generated_tokens
            tbt_sum += batch_tbt_avg
            num_iterations += 1

        avg_tbt = tbt_sum / num_iterations

        # group plotting points into the group_dicts
        bmark_param_match_found = False
        for bmark_param_group_dict in bmark_param_group_dicts:
            if (plotting_knob != 'model_size' and
                bmark_param_group_dict['model_size'] != model_size):
                continue
            if (plotting_knob != 'batch_size' and
                bmark_param_group_dict['batch_size'] != batch_size):
                continue
            if (plotting_knob != 'max_sequence_length' and
                bmark_param_group_dict['max_sequence_length'] != max_sequence_length):
                continue
            if (plotting_knob != 'gpu_type' and
                bmark_param_group_dict['gpu_type'] != gpu_type):
                continue

            # Only reach this point if a match is found
            bmark_param_match_found = True
            bmark_param_group_dict['avg_tbt'].append(avg_tbt)
            break

        # For each bmark_entry, should at least match to one of the plotting groups
        assert(bmark_param_match_found)


# Calcaulates TPS (Tokens Per Second) for different bmark data points and adds information to plotting dicts
def calculate_avg_tps(
    bmark_entries,
    bmark_param_groups,
    excluded_tokens,
    plotting_knob,
    bmark_param_group_dicts
):
    for bmark_entry in bmark_entries:
        model_size = bmark_entry['model_size']
        batch_size = bmark_entry['batch_size']
        max_sequence_length = bmark_entry['max_sequence_length']
        gpu_type = bmark_entry['gpu_type']
        bmark_info = bmark_entry['bmark_info']

        # tps = tokens per second (throughput)
        tps_sum = 0
        # Manually track the number of parsed iterations for edge cases
        num_iterations = 0

        for batch_iteration, batch_dict in bmark_info.items():
            batch_start_time = batch_dict['batch_start_time']
            batch_end_time = batch_dict['batch_end_time']
            batch_e2e_time = batch_end_time - batch_start_time

            batch_input_lengths_sum, batch_output_lengths_sum = 0, 0
            batch_input_tokens_items = batch_dict['batch_input_tokens'].items()
            batch_output_tokens_items = batch_dict['batch_output_tokens'].items()
            assert(len(batch_input_tokens_items) == len(batch_output_tokens_items))

            for (batch_input_tokens_index, batch_input_tokens), (batch_output_tokens_index, batch_output_tokens) in zip(batch_input_tokens_items, batch_output_tokens_items):
                batch_input_length, batch_output_length = 0, 0
                for token in batch_input_tokens:
                    if token not in excluded_tokens:
                        batch_input_length += 1
                for token in batch_output_tokens:
                    if token not in excluded_tokens:
                        batch_output_length += 1

                batch_input_lengths_sum += batch_input_length
                batch_output_lengths_sum += batch_output_length

            assert(batch_size == len(batch_input_tokens_items))
            total_batch_generated_tokens = batch_output_lengths_sum - batch_input_lengths_sum
            # if no tokens generated, skip this iteration
            if total_batch_generated_tokens == 0:
                continue

            # tps = tokens per second
            batch_tps_avg = total_batch_generated_tokens / batch_e2e_time
            tps_sum += batch_tps_avg
            num_iterations += 1

        avg_tps = tps_sum / num_iterations

        # group plotting points into the group_dicts
        bmark_param_match_found = False
        for bmark_param_group_dict in bmark_param_group_dicts:
            if (plotting_knob != 'model_size' and
                bmark_param_group_dict['model_size'] != model_size):
                continue
            if (plotting_knob != 'batch_size' and
                bmark_param_group_dict['batch_size'] != batch_size):
                continue
            if (plotting_knob != 'max_sequence_length' and
                bmark_param_group_dict['max_sequence_length'] != max_sequence_length):
                continue
            if (plotting_knob != 'gpu_type' and
                bmark_param_group_dict['gpu_type'] != gpu_type):
                continue

            # Only reach this point if a match is found
            bmark_param_match_found = True
            bmark_param_group_dict['avg_tps'].append(avg_tps)
            break

        # For each bmark_entry, should at least match to one of the plotting groups
        assert(bmark_param_match_found)


def calculate_avg_ept(
    bmark_entries,
    bmark_param_groups,
    plotting_knob,
    bmark_param_group_dicts,
    gpu_idx
):
    for bmark_entry in bmark_entries:
        model_size = bmark_entry['model_size']
        batch_size = bmark_entry['batch_size']
        max_sequence_length = bmark_entry['max_sequence_length']
        gpu_type = bmark_entry['gpu_type']
        bmark_info = bmark_entry['bmark_info']

        # Extract timestamps from bmark_info
        batch_start_times, batch_end_times = [], []
        for batch_iteration, batch_dict in bmark_info.items():
            batch_start_times.append(batch_dict['batch_start_time'])
            batch_end_times.append(batch_dict['batch_end_time'])

        # Extract timestamps and power usage from nvsmi_info
        nvsmi_info = bmark_entry['nvsmi_info']
        nvsmi_timestamps, nvsmi_curr_powers = [], []
        for nvsmi_dict in nvsmi_info:
            nvsmi_timestamps.append(nvsmi_dict['timestamp_raw'])
            # nvidia-smi monitor records information for all GPUs, even if unused
            # get information about the GPU we actually used for inference
            gpu_idx_dict = nvsmi_dict[gpu_idx]
            nvsmi_curr_powers.append(gpu_idx_dict['curr_power_usage'])

        # For each bmark_entry, calculate the average energy per token across the bmark
        # Average EPT across every iteration in the bmark
        for i in range(len(batch_start_times)):
            batch_start_time, batch_end_time = batch_start_times[i], batch_end_times[i]
            
            # Find the nvsmi timestamps and power metrics that correspond to this batch
            # - First nvsmi timestamp before batch starts
            # - First nvsmi timestamp after batch ends
            nvsmi_before_ts, nvsmi_after_ts = -1, -1
            for j in range(len(nvsmi_timestamps) - 1):
                nvsmi_timestamp_0, nvsmi_timestamp_1 = nvsmi_timestamps[j], nvsmi_timestamps[j+1]
                nvsmi_curr_power_0, nvsmi_curr_power_1 = nvsmi_curr_powers[j], nvsmi_curr_powers[j+1]

                if (nvsmi_timestamp_0 <= batch_start_time and
                    nvsmi_timestamp_1 >= batch_start_time):
                    nvsmi_before_ts = nvsmi_timestamp_0
                if (nvsmi_timestamp_0 <= batch_end_time and
                    nvsmi_timestamp_1 >= batch_end_time):
                    nvsmi_after_ts = nvsmi_timestamp_1

            # sanity checks
            assert(nvsmi_before_ts != -1 and nvsmi_after_ts != -1)
            assert(nvsmi_before_ts <= batch_start_time and
                   batch_start_time <= batch_end_time and
                   batch_end_time <= nvsmi_after_ts)
            #print(f'nvsmi_before_ts: {nvsmi_before_ts}, batch_start_time: {batch_start_time}, batch_end_time: {batch_end_time}, nvsmi_after_ts: {nvsmi_after_ts}')

            # Populate all the nvsmi timestamps and power measurements for the bmark
            batch_nvsmi_timestamps, batch_nvsmi_curr_powers = [], []
            for j in range(len(nvsmi_timestamps)):
                nvsmi_timestamp = nvsmi_timestamps[j]
                nvsmi_curr_power = nvsmi_curr_powers[j]

                if (nvsmi_timestamp >= nvsmi_before_ts and
                    nvsmi_timestamp <= nvsmi_after_ts):
                    batch_nvsmi_timestamps.append(nvsmi_timestamp)
                    batch_nvsmi_curr_powers.append(nvsmi_curr_power)

        ## Make the nvsmi timestamp entries start in the same place as the bmark timestamp entries
        #new_nvsmi_timestamps, new_nvsmi_curr_powers, new_nvsmi_max_powers = [], [], []
        #initial_bmark_timestamp = bmark_timestamps[0]
        #last_bmark_timestamp = bmark_timestamps[-1]
        #for i in range(len(nvsmi_timestamps)):
        #    # leave a 10 second buffer
        #    #if (initial_bmark_timestamp - nvsmi_timestamps[i]) > 10:
        #    #    continue

        #    # only include nvsmi timestamps that are contained within the batch mark timestamps
        #    if (nvsmi_timestamps[i] < initial_bmark_timestamp or
        #        nvsmi_timestamps[i] > last_bmark_timestamp):
        #        continue

        #    # if close enough, then add to plotting lists
        #    new_nvsmi_timestamps.append(nvsmi_timestamps[i])
        #    new_nvsmi_curr_powers.append(nvsmi_curr_powers[i])
        #    new_nvsmi_max_powers.append(nvsmi_max_powers[i])

        #initial_timestamp = new_nvsmi_timestamps[0]
        #for i in range(len(new_nvsmi_timestamps)):
        #    new_nvsmi_timestamps[i] = new_nvsmi_timestamps[i] - initial_timestamp

        #if plot_token_energy:
        #    # calculate all tokens computed during the entire benchmark
        #    total_bmark_generated_tokens = 0

        #    for batch_iteration, batch_dict in bmark_info.items():
        #        batch_input_lengths_sum, batch_output_lengths_sum = 0, 0
        #        for batch_input_length_index, batch_input_lengths in batch_dict['batch_input_lengths'].items():
        #            batch_input_lengths_sum += batch_input_lengths
        #        for batch_output_length_index, batch_output_lengths in batch_dict['batch_output_lengths'].items():
        #            batch_output_lengths_sum += batch_output_lengths
        #        total_batch_generated_tokens = batch_output_lengths_sum - batch_input_lengths_sum
        #        total_bmark_generated_tokens += total_batch_generated_tokens

        #    # calculate the total energy consumed during the entire benchmark
        #    total_bmark_joules = np.trapz(new_nvsmi_curr_powers, new_nvsmi_timestamps)
        #    joules_per_token = total_bmark_joules / total_bmark_generated_tokens


def plot_ept_vs_tbt(
    bmark_entries,
    bmark_param_groups,
    excluded_tokens,
    gpu_idx,
    plot_filename,
    plot_name
):
    # Organizing different bmark data points for the line plot
    plotting_metrics = [
        'batch_size',
        'avg_ept', # EPT: Energy Per Token
        'avg_tbt'  # TBT: Time Between Tokens
    ]
    bmark_param_group_dicts = group_experiment_data(
        bmark_entries,
        bmark_param_groups,
        plotting_metrics
    )
    plotting_knob = 'batch_size'

    # Populate bmark_param_group_dicts with the plotting knob lists
    # For this graph, batch_size is the plotting knob
    for bmark_entry in bmark_entries:
        model_size = bmark_entry['model_size']
        batch_size = bmark_entry['batch_size']
        max_sequence_length = bmark_entry['max_sequence_length']
        gpu_type = bmark_entry['gpu_type']
        bmark_info = bmark_entry['bmark_info']

        # group plotting points into the group_dicts
        # populate plotting knob field with all the different data points
        bmark_param_match_found = False
        for bmark_param_group_dict in bmark_param_group_dicts:
            if (plotting_knob != 'model_size' and
                bmark_param_group_dict['model_size'] != model_size):
                continue
            if (plotting_knob != 'batch_size' and
                bmark_param_group_dict['batch_size'] != batch_size):
                continue
            if (plotting_knob != 'max_sequence_length' and
                bmark_param_group_dict['max_sequence_length'] != max_sequence_length):
                continue
            if (plotting_knob != 'gpu_type' and
                bmark_param_group_dict['gpu_type'] != gpu_type):
                continue

            # Only reach this point if a match is found
            bmark_param_match_found = True
            bmark_param_group_dict[plotting_knob].append(batch_size)
            break

        # For each bmark_entry, should at least match to one of the plotting groups
        assert(bmark_param_match_found)

    # Calculate TBT
    calculate_avg_tbt(
        bmark_entries,
        bmark_param_groups,
        excluded_tokens,
        plotting_knob,
        bmark_param_group_dicts
    )
    # Calculate EPT
    calculate_avg_ept(
        bmark_entries,
        bmark_param_groups,
        plotting_knob,
        bmark_param_group_dicts,
        gpu_idx
    )


# TODO: - This is theoretical user-perceived latency to provide a bound for tbt (time between tokens).
#       - Actual user-perceived latency depends on how quickly the new tokens actually make it to the user.
#       - TTFT (time to first token) is also an important metric, but is not taken into account with these experiments.
# throughput : tokens per second
# latency    : theoretical user-perceived seconds per token (tbt)
def plot_tps_vs_tbt(
    bmark_entries,
    bmark_param_groups,
    excluded_tokens,
    plot_filename,
    plot_name
):
    # Organizing different bmark data points for the line plot
    plotting_metrics = [
        'batch_size',
        'avg_tps', # TPS: Tokens Per Second
        'avg_tbt'  # TBT: Time Between Tokens
    ]
    bmark_param_group_dicts = group_experiment_data(
        bmark_entries,
        bmark_param_groups,
        plotting_metrics
    )
    plotting_knob = 'batch_size'

    # Populate bmark_param_group_dicts with the plotting knob lists
    # For this graph, batch_size is the plotting knob
    for bmark_entry in bmark_entries:
        model_size = bmark_entry['model_size']
        batch_size = bmark_entry['batch_size']
        max_sequence_length = bmark_entry['max_sequence_length']
        gpu_type = bmark_entry['gpu_type']
        bmark_info = bmark_entry['bmark_info']

        # group plotting points into the group_dicts
        # populate plotting knob field with all the different data points
        bmark_param_match_found = False
        for bmark_param_group_dict in bmark_param_group_dicts:
            if (plotting_knob != 'model_size' and
                bmark_param_group_dict['model_size'] != model_size):
                continue
            if (plotting_knob != 'batch_size' and
                bmark_param_group_dict['batch_size'] != batch_size):
                continue
            if (plotting_knob != 'max_sequence_length' and
                bmark_param_group_dict['max_sequence_length'] != max_sequence_length):
                continue
            if (plotting_knob != 'gpu_type' and
                bmark_param_group_dict['gpu_type'] != gpu_type):
                continue

            # Only reach this point if a match is found
            bmark_param_match_found = True
            bmark_param_group_dict[plotting_knob].append(batch_size)
            break

        # For each bmark_entry, should at least match to one of the plotting groups
        assert(bmark_param_match_found)

    # Calculate TBT
    calculate_avg_tbt(
        bmark_entries,
        bmark_param_groups,
        excluded_tokens,
        plotting_knob,
        bmark_param_group_dicts
    )
    # Calculate TPS
    calculate_avg_tps(
        bmark_entries,
        bmark_param_groups,
        excluded_tokens,
        plotting_knob,
        bmark_param_group_dicts
    )

    plt.figure(figsize=(8, 3))
    for bmark_param_group_dict in bmark_param_group_dicts:
        for key, val in bmark_param_group_dict.items():
            print(f'{key}, {val}')

        avg_tps = bmark_param_group_dict['avg_tps']
        avg_tbt = bmark_param_group_dict['avg_tbt']
        batch_sizes = bmark_param_group_dict['batch_size']

        model_size = bmark_param_group_dict['model_size']
        max_sequence_length = bmark_param_group_dict['max_sequence_length']
        gpu_type = bmark_param_group_dict['gpu_type']
        plt.plot(avg_tps, avg_tbt, label=f'{model_size} {gpu_type}', marker='o')

        for avg_tps_val, avg_tbt_val, batch_size in zip(avg_tps, avg_tbt, batch_sizes):
            plt.annotate(str(batch_size),
                         (avg_tps_val, avg_tbt_val),
                         textcoords='offset points',
                         xytext=(0, 10),
                         ha='center')

    plt.xlabel('Tokens Per Second')
    plt.ylabel('Avg. Request Token Latency')
    plt.title(plot_name)
    plt.grid(True)
    legend = plt.legend()
    legend._legend_box.sep = 3
    legend._legend_box.align = "right"
    plt.setp(legend.get_texts(), fontsize='small')
    plt.setp(legend.get_patches(), scalex=0.5, scaley=0.5)
    plt.tight_layout()
    plt.savefig('plots/' + plot_filename)


# TODO: I think this has been haphazardly converted to a GPU utilization plotter
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
    plt.savefig('plots/' + 'v10032gb_llama7b_utilization.png')

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

#def joules_to_pittsburgh_carbon(joules):
#    pittsburgh_carbon_intensity = 413
#
#    kWh = joules / 3600000
#    grams_co2_eq = kWh * pittsburgh_carbon_intensity
#    return grams_co2_eq






#def plot_average_batch_latency(
#    bmark_entries,
#    plot_filename,
#    plot_sequence_lengths,
#    plot_batch_sizes,
#    bmark_param_groups
#):
#    plt.figure(figsize=(10, 3))
#
#    bmark_param_group_dicts = []
#    for bmark_param_group in bmark_param_groups:
#        group_split = bmark_param_group.split()
#        bmark_param_group_dict = {}
#        bmark_param_group_dict['model_size'] = int(group_split[0]) if group_split[0] != 'X' else 'X'
#        bmark_param_group_dict['batch_size'] = int(group_split[1]) if group_split[1] != 'X' else 'X'
#        bmark_param_group_dict['max_sequence_length'] = int(group_split[2]) if group_split[2] != 'X' else 'X'
#        bmark_param_group_dict['avg_batch_latencies'] = []
#        bmark_param_group_dicts.append(bmark_param_group_dict)
#
#    for bmark_entry in bmark_entries:
#        model_size = bmark_entry['model_size']
#        batch_size = bmark_entry['batch_size']
#        max_sequence_length = bmark_entry['max_sequence_length']
#        if max_sequence_length not in plot_sequence_lengths:
#            continue
#        if batch_size not in plot_batch_sizes:
#            continue
#
#        # Extract timestamps from bmark_info
#        bmark_info = bmark_entry['bmark_info']
#        # each entry is (batch_start_time, batch_end_time)
#        bmark_tuples = []
#        curr_max_time = 0.0
#        batch_latency_sum = 0.0
#        num_iterations = len(bmark_info)
#
#        for batch_iteration, batch_dict in bmark_info.items():
#            batch_start_time = batch_dict['batch_start_time']
#            batch_end_time = batch_dict['batch_end_time']
#
#            # make sure timestamps are strictly increasing
#            assert(batch_start_time > curr_max_time and
#                   batch_end_time > batch_start_time)
#            curr_max_time = batch_end_time
#            batch_latency = batch_end_time - batch_start_time
#            batch_latency_sum += batch_latency
#
#        avg_batch_latency = batch_latency_sum / num_iterations
#        print(f'PLOT_AVERAGE_BATCH_LATENCY model_size: {model_size}, batch_size: {batch_size}, max_sequence_length: {max_sequence_length}, avg_batch_latency: {avg_batch_latency}, batch_size: {batch_size}')
#        avg_batch_latencies_dict = {
#            'batch_size': batch_size,
#            'avg_batch_latency': avg_batch_latency
#        }
#
#        # Identify which bmark_param_group_dict to append to
#        for bmark_param_group_dict in bmark_param_group_dicts:
#            bmark_param_match_found = False
#            if (bmark_param_group_dict['model_size'] != 'X' and
#                bmark_param_group_dict['model_size'] != model_size):
#                continue
#            if (bmark_param_group_dict['batch_size'] != 'X' and
#                bmark_param_group_dict['batch_size'] != batch_size):
#                continue
#            if (bmark_param_group_dict['max_sequence_length'] != 'X' and
#                bmark_param_group_dict['max_sequence_length'] != max_sequence_length):
#                continue
#
#            # Only reach this point if a match is found
#            bmark_param_match_found = True
#            bmark_param_group_dict['avg_batch_latencies'].append(avg_batch_latencies_dict)
#            break
#
#        # For each bmark_entry, should at least match to one of the plotting groups
#        assert(bmark_param_match_found)
#
#    for bmark_param_group_dict in bmark_param_group_dicts:
#        batch_sizes = []
#        avg_latencies = []
#        for avg_batch_latencies_dict in bmark_param_group_dict['avg_batch_latencies']:
#            batch_sizes.append(avg_batch_latencies_dict['batch_size'])
#            avg_latencies.append(avg_batch_latencies_dict['avg_batch_latency'])
#
#        plt.plot(batch_sizes, avg_latencies, label=f'Llama {bmark_param_group_dict["model_size"]}B Max {bmark_param_group_dict["max_sequence_length"]} Sequence Length)')
#
#    # Minimum batch size of 1
#    plt.xlim(left=1)
#    plt.xlabel('batch size')
#    plt.ylabel('Time (seconds)')
#    plt.title(f'Avg. Batch Latency vs. Batch Size')
#    plt.legend()
#    plt.grid(True)
#    plt.tight_layout()
#    plt.savefig('plots/' + plot_filename)



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
        model_size = curr_bmark_params[0]
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

    if args.plot_tps_vs_tbt:
        plot_tps_vs_tbt(
            bmark_entries,
            args.bmark_param_groups,
            args.excluded_tokens,
            args.plot_filename,
            args.plot_name
        )
    if args.plot_ept_vs_tbt:
        plot_ept_vs_tbt(
            bmark_entries,
            args.bmark_param_groups,
            args.excluded_tokens,
            args.gpu_idx,
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
        '--plot_tps_vs_tbt',
        default=False,
        action='store_true',
        help='specify this arg to plot tps (tokens per second) vs tbt (time between tokens)'
    )
    parser.add_argument(
        '--plot_ept_vs_tbt',
        default=False,
        action='store_true',
        help='specify this arg to plot ept (energy per token) vs tbt (time between tokens)'
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
