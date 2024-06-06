import argparse
import re
import ast
import matplotlib.pyplot as plt
import math
from pathlib import Path

import numpy as np
#import transformer_model_scaling
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


# TODO: - This is theoretical user-perceived latency to provide a bound for tbt (time between tokens).
#       - Actual user-perceived latency depends on how quickly the new tokens actually make it to the user.
#       - TTFT (time to first token) is also an important metric, but is not taken into account with these experiments.
# throughput : tokens per second
# latency    : theoretical user-perceived seconds per token (tbt)
def plot_throughput_vs_tbt(
    bmark_entries,
    bmark_param_groups,
    plot_filename,
    plot_name
):
    # Organizing different bmark data points for the line plot
    # For latency vs. throughput plots, track batch_sizes + avg tps + avg spt
    plotting_metrics = [
        'batch_sizes',
        'avg_tpss',
        'avg_spts',
        'avg_batch_e2e_times'
    ]
    bmark_param_group_dicts = group_experiment_data(
        bmark_entries,
        bmark_param_groups,
        plotting_metrics
    )

    for bmark_entry in bmark_entries:
        model_size = bmark_entry['model_size']
        batch_size = bmark_entry['batch_size']
        max_sequence_length = bmark_entry['max_sequence_length']
        gpu_type = bmark_entry['gpu_type']
        bmark_info = bmark_entry['bmark_info']
        print(f'bmark_entry: {model_size} {batch_size} {max_sequence_length} {gpu_type}')

        # tps = tokens per second (throughput)
        tps_sum = 0
        # tbt = time between tokens (theoretically achievable user-perceived latency)
        tbt_sum = 0
        e2e_time_sum = 0
        num_iterations = 0

        for batch_iteration, batch_dict in bmark_info.items():
            batch_start_time = batch_dict['batch_start_time']
            batch_end_time = batch_dict['batch_end_time']
            batch_e2e_time = batch_end_time - batch_start_time

            batch_input_lengths_sum, batch_output_lengths_sum, batch_tbt_sum = 0, 0, 0
            batch_input_lengths_items = batch_dict['batch_input_lengths'].items()
            batch_output_lengths_items = batch_dict['batch_output_lengths'].items()
            assert(len(batch_input_lengths_items) == len(batch_output_lengths_items))

            for i in range(len(batch_input_lengths_items)):
                batch_input_length_index, batch_input_length = batch_input_lengths_items[i]
                batch_output_length_index, batch_output_length = batch_output_lengths_items[i]
                batch_tbt = batch_e2e_time / (batch_output_length - batch_input_length)

                batch_input_lengths_sum += batch_input_length
                batch_output_lengths_sum += batch_output_length
                batch_tbt_sum += batch_tbt

            assert(batch_size == len(batch_input_lengths_items))
            total_batch_generated_tokens = batch_output_lengths_sum - batch_input_lengths_sum
            batch_tbt_avg = batch_tbt_sum / batch_size

            # In the off case that there were no generated tokens in this batch, skip this iteration
            if avg_batch_generated_tokens == 0:
                continue

            # tps = tokens per second
            batch_tps_avg = total_batch_generated_tokens / batch_e2e_time
            tps_sum += batch_tps_avg
            tbt_sum += batch_tbt_avg
            e2e_time_sum += batch_e2e_time
            num_iterations += 1

        avg_tps = tps_sum / num_iterations
        avg_tbt = tbt_sum / num_iterations
        print(f'{model_size} {batch_size} {max_sequence_length} {gpu_type} {avg_tps} {avg_tbt}')

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
            bmark_param_group_dict['avg_tbts'].append(avg_tbt)
            bmark_param_group_dict['avg_batch_e2e_times'].append(avg_batch_e2e_time)
            break

        # For each bmark_entry, should at least match to one of the plotting groups
        assert(bmark_param_match_found)

    plt.figure(figsize=(8, 3))
    for bmark_param_group_dict in bmark_param_group_dicts:
        for key, val in bmark_param_group_dict.items():
            print(f'{key}, {val}')

        avg_tpss = bmark_param_group_dict['avg_tpss']
        avg_tbts = bmark_param_group_dict['avg_tbts']
        avg_batch_e2e_times = bmark_param_group_dict['avg_batch_e2e_times']
        batch_sizes = bmark_param_group_dict['batch_sizes']

        model_size = bmark_param_group_dict['model_size']
        max_sequence_length = bmark_param_group_dict['max_sequence_length']
        gpu_type = bmark_param_group_dict['gpu_type']
        plt.plot(avg_tpss, avg_tbts, label=f'{model_size} {gpu_type}', marker='o')

        for avg_tps, avg_tbt, batch_size in zip(avg_tpss, avg_tbts, batch_sizes):
            plt.annotate(str(batch_size),
                         (avg_tps, avg_tbt),
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

def joules_to_pittsburgh_carbon(joules):
    pittsburgh_carbon_intensity = 413

    kWh = joules / 3600000
    grams_co2_eq = kWh * pittsburgh_carbon_intensity
    return grams_co2_eq


def plot_power_or_energy(
    bmark_entries,
    plot_filename,
    plot_name,
    gpu_idx,
    project_24_hr,
    plot_token_energy
):
    plot_a100_max_power = False
    plot_v100_max_power = False

    plt.figure(figsize=(10, 4))

    joules_per_token_list = []
    joules_label_list = []

    for bmark_entry in bmark_entries:
        model_size = bmark_entry['model_size']
        batch_size = bmark_entry['batch_size']
        max_sequence_length = bmark_entry['max_sequence_length']
        gpu_type = bmark_entry['gpu_type']
        print(f'bmark_entry: {model_size} {batch_size} {max_sequence_length} {gpu_type}')

        if gpu_type == 'a10040gb':
            plot_a100_max_power = True
        if gpu_type == 'v10032gb':
            plot_v100_max_power = True

        # Extract timestamps from bmark_info
        bmark_info = bmark_entry['bmark_info']
        # each entry is (batch_start_time, batch_end_time)
        bmark_timestamps = []
        for batch_iteration, batch_dict in bmark_info.items():
            batch_start_time = batch_dict['batch_start_time']
            batch_end_time = batch_dict['batch_end_time']
            bmark_timestamps.append(batch_start_time)
            bmark_timestamps.append(batch_end_time)
        bmark_entry['bmark_timestamps'] = bmark_timestamps

        bmark_info = bmark_entry['bmark_info']
        # Extract timestamps and power usage from nvsmi_info
        nvsmi_info = bmark_entry['nvsmi_info']
        # each entry is (timestamp_raw, curr_power_usage, max_power_usage)
        nvsmi_timestamps = []
        nvsmi_curr_powers = []
        nvsmi_max_powers = []
        for nvsmi_dict in nvsmi_info:
            timestamp_raw = nvsmi_dict['timestamp_raw']
            nvsmi_timestamps.append(timestamp_raw)

            gpu_idx_dict = nvsmi_dict[gpu_idx]
            curr_power_usage = gpu_idx_dict['curr_power_usage']
            max_power_usage = gpu_idx_dict['max_power_usage']
            nvsmi_curr_powers.append(curr_power_usage)
            nvsmi_max_powers.append(max_power_usage)

        # Make the nvsmi timestamp entries start in the same place as the bmark timestamp entries
        assert(len(nvsmi_timestamps) == len(nvsmi_curr_powers) and
               len(nvsmi_curr_powers) == len(nvsmi_max_powers))
        new_nvsmi_timestamps, new_nvsmi_curr_powers, new_nvsmi_max_powers = [], [], []
        initial_bmark_timestamp = bmark_timestamps[0]
        last_bmark_timestamp = bmark_timestamps[-1]
        for i in range(len(nvsmi_timestamps)):
            # leave a 10 second buffer
            #if (initial_bmark_timestamp - nvsmi_timestamps[i]) > 10:
            #    continue

            # only include nvsmi timestamps that are contained within the batch mark timestamps
            if (nvsmi_timestamps[i] < initial_bmark_timestamp or
                nvsmi_timestamps[i] > last_bmark_timestamp):
                continue

            # if close enough, then add to plotting lists
            new_nvsmi_timestamps.append(nvsmi_timestamps[i])
            new_nvsmi_curr_powers.append(nvsmi_curr_powers[i])
            new_nvsmi_max_powers.append(nvsmi_max_powers[i])

        initial_timestamp = new_nvsmi_timestamps[0]
        for i in range(len(new_nvsmi_timestamps)):
            new_nvsmi_timestamps[i] = new_nvsmi_timestamps[i] - initial_timestamp

        if plot_token_energy:
            # calculate all tokens computed during the entire benchmark
            total_bmark_generated_tokens = 0

            for batch_iteration, batch_dict in bmark_info.items():
                batch_input_lengths_sum, batch_output_lengths_sum = 0, 0
                for batch_input_length_index, batch_input_lengths in batch_dict['batch_input_lengths'].items():
                    batch_input_lengths_sum += batch_input_lengths
                for batch_output_length_index, batch_output_lengths in batch_dict['batch_output_lengths'].items():
                    batch_output_lengths_sum += batch_output_lengths
                total_batch_generated_tokens = batch_output_lengths_sum - batch_input_lengths_sum
                total_bmark_generated_tokens += total_batch_generated_tokens

            # calculate the total energy consumed during the entire benchmark
            total_bmark_joules = np.trapz(new_nvsmi_curr_powers, new_nvsmi_timestamps)
            joules_per_token = total_bmark_joules / total_bmark_generated_tokens

            joules_per_token_list.append(joules_per_token)
            joules_label_list.append(f'{model_size} {batch_size} {gpu_type}')


        # project power measurements over a 24 hour period
        if project_24_hr:
            # take the inner slice of timestamps and power
            timestamp_slice = new_nvsmi_timestamps[15:-15].copy()
            curr_powers_slice = new_nvsmi_curr_powers[15:-15].copy()
            timestamp_slice_copy = timestamp_slice.copy()
            curr_powers_slice_copy = curr_powers_slice.copy()
            timestamp_offset = timestamp_slice[-1] + 1

            # Take the average of the curr_powers_slice
            curr_powers_sum = 0
            for curr_power in curr_powers_slice:
                curr_powers_sum += curr_power
            curr_powers_avg = curr_powers_sum / len(curr_powers_slice)
            curr_powers_avg_portion = 0.25 * curr_powers_avg
            lower_bound = curr_powers_avg - curr_powers_avg_portion
            upper_bound = curr_powers_avg + curr_powers_avg_portion

            timestamp_slice_duration = timestamp_slice[-1] - timestamp_slice[0]
            print(f'timestamp_slice_duration: {timestamp_slice_duration}')
            # find how many slices fit into a day (ceil)
            slice_multiplier = math.ceil((24 * 60 * 60) / timestamp_slice_duration)
            print(f'slice_multiplier: {slice_multiplier}')

            # append slices until 24 hours is reached
            for j in range(slice_multiplier):
                # add offset to timestamp_slice_copy
                for k in range(len(timestamp_slice_copy)):
                    timestamp_slice_copy[k] += timestamp_offset

                for i in range(len(timestamp_slice_copy)):
                    timestamp_slice.append(timestamp_slice_copy[i])
                    curr_powers_slice.append(curr_powers_slice_copy[i])

            # cut off entries that go past 24 hours
            new_nvsmi_timestamps = []
            new_nvsmi_curr_powers = []
            assert(len(timestamp_slice) == len(curr_powers_slice))
            for j in range(len(timestamp_slice)):
                # only show 5 minutes of power trace to make plot legible
                if timestamp_slice[j] > (10 * 60):
                    break
                new_nvsmi_timestamps.append(timestamp_slice[j])

                # avg-based smoothing so that the plot is legible
                if (curr_powers_slice[j] < lower_bound or
                    curr_powers_slice[j] > upper_bound):
                    new_nvsmi_curr_powers.append(curr_powers_avg)
                else:
                    new_nvsmi_curr_powers.append(curr_powers_slice[j])
                #new_nvsmi_curr_powers.append(curr_powers_slice[j])

            # no groupings necessary for these plots
            print(f'new_nvsmi_timestamps: {new_nvsmi_timestamps}')
            print(f'new_nvsmi_curr_powers: {new_nvsmi_curr_powers}')
            plt.plot(new_nvsmi_timestamps, new_nvsmi_curr_powers, label=f'{model_size} {batch_size} {gpu_type}')

    if plot_token_energy:
        gcarb_per_token_list = []
        for jtok in joules_per_token_list:
            gcarb_per_token_list.append(joules_to_pittsburgh_carbon(jtok))

        #plt.bar(range(len(joules_per_token_list)), joules_per_token_list, tick_label=joules_label_list)
        plt.bar(range(len(gcarb_per_token_list)), gcarb_per_token_list, tick_label=joules_label_list)
        plt.xlabel('Data Point')
        plt.ylabel('Grams Carbon Per Token')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.grid(True)
        plt.title(plot_name)
        #plt.legend()
        plt.savefig('plots/' + plot_filename)

    if project_24_hr:
        #if plot_a100_max_power:
        #    plt.axhline(y=400, color='red', linestyle='--', label='Peak A100 Power')
        if plot_v100_max_power:
            plt.axhline(y=250, color='orange', linestyle='--', label='Peak V100 Power')
        plt.xlabel('Time (seconds)')
        plt.ylabel('Power Usage (W)')
        plt.title(plot_name)
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        #plt.ylim(100, 260)
        plt.savefig('plots/' + plot_filename)


def plot_average_batch_latency(
    bmark_entries,
    plot_filename,
    plot_sequence_lengths,
    plot_batch_sizes,
    bmark_param_groups
):
    plt.figure(figsize=(10, 3))

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
    plt.savefig('plots/' + plot_filename)



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

    if args.plot_power_or_energy:
        plot_power_or_energy(
            bmark_entries,
            args.plot_filename,
            args.plot_name,
            args.gpu_idx,
            args.project_24_hr,
            args.plot_token_energy
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
        plot_normalized_token_latency(
            bmark_entries,
            args.plot_filename,
            args.bmark_param_groups,
            args.excluded_tokens
        )
    if args.plot_throughput_vs_tbt:
        plot_throughput_vs_tbt(
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
        '--plot_power_or_energy',
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
        '--plot_throughput_vs_tbt',
        default=False,
        action='store_true',
        help='specify this arg to plot throughput vs tbt (time between tokens)'
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
    parser.add_argument(
        '--project_24_hr',
        default=False,
        action='store_true',
        help='specify this for power plots to project power measurements over a 24 hour period'
    )
    parser.add_argument(
        '--plot_token_energy',
        default=False,
        action='store_true',
        help='specify this for energy-per-token plots of the provided data points'
    )
    args = parser.parse_args()
    main(args)
