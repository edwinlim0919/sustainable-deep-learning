import argparse
import re
import ast
import matplotlib.pyplot as plt
import numpy as np
import math
from pathlib import Path

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
        bmark_param_group_dict['gpu_type'] = group_split[2] if group_split[2] != 'X' else 'X'

        for plotting_metric in plotting_metrics:
            bmark_param_group_dict[plotting_metric] = []
        bmark_param_group_dicts.append(bmark_param_group_dict)

    return bmark_param_group_dicts


# Match data into the right plotting group and update benchmarking info
def update_experiment_data(
    bmark_param_group_dicts,
    plotting_knob,
    bmark_param_group_dict_key,
    bmark_param_group_dict_val,
    bmark_entry
):
    model_size = bmark_entry['model_size']
    batch_size = bmark_entry['batch_size']
    gpu_type = bmark_entry['gpu_type']

    # group plotting points into the group_dicts
    bmark_param_match_found = False
    for bmark_param_group_dict in bmark_param_group_dicts:
        if (plotting_knob != 'model_size' and
            bmark_param_group_dict['model_size'] != model_size):
            continue
        if (plotting_knob != 'batch_size' and
            bmark_param_group_dict['batch_size'] != batch_size):
            continue
        if (plotting_knob != 'gpu_type' and
            bmark_param_group_dict['gpu_type'] != gpu_type):
            continue

        # Only reach this point if a match is found
        bmark_param_match_found = True
        bmark_param_group_dict[bmark_param_group_dict_key].append(bmark_param_group_dict_val)
        break

    # For each bmark_entry, should at least match to one of the plotting groups
    assert(bmark_param_match_found)


# Calcaulates TBI (Time Between Images) for different bmark data points and adds information to plotting dicts
def calculate_avg_tbi(
    bmark_entries,
    bmark_param_groups,
    plotting_knob,
    bmark_param_group_dicts
):
    for bmark_entry in bmark_entries:
        batch_size = bmark_entry['batch_size']
        bmark_info = bmark_entry['bmark_info']

        # For this given bmark data point, keep track of running tbi sum to calculate avg at the end
        tbi_sum = 0
        # Manually track the number of parsed iterations for edge cases
        num_iterations = 0

        for batch_iteration, batch_dict in bmark_info.items():
            batch_start_time = batch_dict['batch_start_time']
            batch_end_time = batch_dict['batch_end_time']
            batch_e2e_time = batch_end_time - batch_start_time
            
            # tbi = time between images
            batch_tbi_avg = batch_e2e_time / batch_size
            tbi_sum += batch_tbi_avg
            num_iterations += 1

        avg_tbi = tbi_sum / num_iterations
        update_experiment_data(
            bmark_param_group_dicts,
            plotting_knob,
            'avg_tbi',
            avg_tbi,
            bmark_entry
        )


# Calcaulates IPS (Images Per Second) for different bmark data points and adds information to plotting dicts
def calculate_avg_ips(
    bmark_entries,
    bmark_param_groups,
    plotting_knob,
    bmark_param_group_dicts
):
    for bmark_entry in bmark_entries:
        batch_size = bmark_entry['batch_size']
        bmark_info = bmark_entry['bmark_info']

        # ips = images per second (throughput)
        ips_sum = 0
        # Manually track the number of parsed iterations for edge cases
        num_iterations = 0

        for batch_iteration, batch_dict in bmark_info.items():
            batch_start_time = batch_dict['batch_start_time']
            batch_end_time = batch_dict['batch_end_time']
            batch_e2e_time = batch_end_time - batch_start_time
            
            # ips = images per second
            batch_ips_avg = batch_size / batch_e2e_time
            ips_sum += batch_ips_avg
            num_iterations += 1

        avg_ips = ips_sum / num_iterations
        update_experiment_data(
            bmark_param_group_dicts,
            plotting_knob,
            'avg_ips',
            avg_ips,
            bmark_entry
        )

# Calculates EPI (Energy Per Image)
def calculate_avg_epi(
    bmark_entries,
    bmark_param_groups,
    plotting_knob,
    gpu_idxs,
    bmark_param_group_dicts
):
    for bmark_entry in bmark_entries:
        batch_size = bmark_entry['batch_size']
        bmark_info = bmark_entry['bmark_info']

        # Extract timestamps from bmark_info
        batch_start_times, batch_end_times = [], []
        # batch_total_images_list = []
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
            nvsmi_curr_power = []
            for gpu_idx in gpu_idxs:
                gpu_idx_dict = nvsmi_dict[gpu_idx]
                nvsmi_curr_power.append(gpu_idx_dict['curr_power_usage'])
            nvsmi_curr_powers.append(nvsmi_curr_power)

        # For this given bmark data point, keep track of running tbi sum to calculate avg at the end
        epi_sum = 0
        # Manually track the number of parsed iterations for edge cases
        num_iterations = 0

        # For each bmark_entry, calculate the average energy per image across the bmark
        # Average EPI across every iteration in the bmark
        for i in range(len(batch_start_times)):
            batch_start_time, batch_end_time = batch_start_times[i], batch_end_times[i]

            # Find the nvsmi timestamps and power metrics that correspond to this batch
            # - First nvsmi timestamp before batch starts
            # - First nvsmi timestamp after batch ends
            nvsmi_before_ts, nvsmi_after_ts = -1, -1
            for j in range(len(nvsmi_timestamps) - 1):
                nvsmi_timestamp_0, nvsmi_timestamp_1 = nvsmi_timestamps[j], nvsmi_timestamps[j+1]
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

            # Populate all the nvsmi timestamps and power measurements for the bmark
            batch_nvsmi_timestamps, batch_nvsmi_curr_powers = [], []
            for j in range(len(nvsmi_timestamps)):
                nvsmi_timestamp = nvsmi_timestamps[j]
                nvsmi_curr_power = nvsmi_curr_powers[j]

                if (nvsmi_timestamp >= nvsmi_before_ts and
                    nvsmi_timestamp <= nvsmi_after_ts):
                    batch_nvsmi_timestamps.append(nvsmi_timestamp)
                    batch_nvsmi_curr_powers.append(nvsmi_curr_power)

            # For calculating energy (area under curve), calculate energy purely for when the batch is being computed
            # So replace first and last recorded nvsmi timestamps with the start and end time of the batch
            batch_nvsmi_timestamps[0] = batch_start_time
            batch_nvsmi_timestamps[-1] = batch_end_time

            # w/ 4 GPUs batch_nvsmi_curr_powers looks like [[idx0, idx1, idx2, idx3] ... [idx0, idx1, idx2, idx3]]
            # to calculate energy/area under curve, batch_nvsmi_curr_powers_split needs to look like [[idx0 ... idx0] ... [idx3 ... idx3]]
            batch_nvsmi_curr_powers_split = []
            for j in range(len(gpu_idxs)):
                batch_nvsmi_curr_powers_split.append([])
            for batch_nvsmi_curr_power in batch_nvsmi_curr_powers:
                for j in range(len(gpu_idxs)):
                    batch_nvsmi_curr_powers_split[j].append(batch_nvsmi_curr_power[j])

            energy_joules_sum = 0
            for j in range(len(gpu_idxs)):
                energy_joules = np.trapz(batch_nvsmi_curr_powers_split[j], batch_nvsmi_timestamps)
                energy_joules_sum += energy_joules

            batch_epi_avg = energy_joules_sum / batch_size
            epi_sum += batch_epi_avg
            num_iterations += 1

        avg_epi = epi_sum / num_iterations
        update_experiment_data(
            bmark_param_group_dicts,
            plotting_knob,
            'avg_epi',
            avg_epi,
            bmark_entry
        )

def plot_tbi_vs_epi(
    bmark_entries,
    bmark_param_groups,
    gpu_idxs,
    plot_filename,
    plot_name
):
    # Organizing different bmark data points for the line plot
    plotting_metrics = [
        'batch_size',
        'avg_epi', # EPI: Energy Per Image
        'avg_tbi'  # TBI: Time Between Images
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
        batch_size = bmark_entry['batch_size']
        update_experiment_data(
            bmark_param_group_dicts,
            plotting_knob,
            plotting_knob,
            batch_size,
            bmark_entry
        )

    # Calculate TBI
    calculate_avg_tbi(
        bmark_entries,
        bmark_param_groups,
        plotting_knob,
        bmark_param_group_dicts
    )
    # Calculate EPI
    calculate_avg_epi(
        bmark_entries,
        bmark_param_groups,
        plotting_knob,
        gpu_idxs,
        bmark_param_group_dicts
    )

    plt.figure(figsize=(8, 3))
    for bmark_param_group_dict in bmark_param_group_dicts:
        for key, val in bmark_param_group_dict.items():
            print(f'{key}: {val}')

        avg_epi = bmark_param_group_dict['avg_epi']
        avg_tbi = bmark_param_group_dict['avg_tbi']
        batch_sizes = bmark_param_group_dict['batch_size']

        model_size = bmark_param_group_dict['model_size']
        gpu_type = bmark_param_group_dict['gpu_type']
        plt.plot(avg_tbi, avg_epi, label=f'{model_size} {gpu_type}', marker='o')

        for avg_tbi_val, avg_epi_val, batch_size in zip(avg_tbi, avg_epi, batch_sizes):
            plt.annotate(str(batch_size),
                         (avg_tbi_val, avg_epi_val),
                         textcoords='offset points',
                         xytext=(0, 10),
                         ha='center')

    plt.xlabel('Time Between images')
    plt.ylabel('Joules Per image')
    plt.title(plot_name)
    plt.grid(True)
    legend = plt.legend()
    legend._legend_box.sep = 3
    legend._legend_box.align = "right"
    plt.setp(legend.get_texts(), fontsize='small')
    plt.setp(legend.get_patches(), scalex=0.5, scaley=0.5)
    plt.tight_layout()
    plt.savefig('plotting/plots/' + plot_filename)


#  - This is theoretical user-perceived latency to provide a bound for tbi (time between images).
#  - Actual user-perceived latency depends on how quickly the new images actually make it to the user.
#  - TTFT (time to first image) is also an important metric, but is not taken into account with these experiments.
# throughput : images per second
# latency    : theoretical user-perceived seconds per image (tbi)
def plot_ips_vs_tbi(
    bmark_entries,
    bmark_param_groups,
    plot_filename,
    plot_name,
    tbi_slo
):
    # Organizing different bmark data points for the line plot
    plotting_metrics = [
        'batch_size',
        'avg_ips', # IPS: Images Per Second
        'avg_tbi'  # TBI: Time Between Images
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
        batch_size = bmark_entry['batch_size']
        update_experiment_data(
            bmark_param_group_dicts,
            plotting_knob,
            plotting_knob,
            batch_size,
            bmark_entry
        )

    # Calculate tbi
    calculate_avg_tbi(
        bmark_entries,
        bmark_param_groups,
        plotting_knob,
        bmark_param_group_dicts
    )
    # Calculate TPS
    calculate_avg_ips(
        bmark_entries,
        bmark_param_groups,
        plotting_knob,
        bmark_param_group_dicts
    )

    plt.figure(figsize=(8, 3))
    for bmark_param_group_dict in bmark_param_group_dicts:
        for key, val in bmark_param_group_dict.items():
            print(f'{key}: {val}')

        avg_ips = bmark_param_group_dict['avg_ips']
        avg_tbi = bmark_param_group_dict['avg_tbi']
        batch_sizes = bmark_param_group_dict['batch_size']

        model_size = bmark_param_group_dict['model_size']
        gpu_type = bmark_param_group_dict['gpu_type']
        plt.plot(avg_ips, avg_tbi, label=f'{model_size} {gpu_type}', marker='o')

        for avg_ips_val, avg_tbi_val, batch_size in zip(avg_ips, avg_tbi, batch_sizes):
            plt.annotate(str(batch_size),
                         (avg_ips_val, avg_tbi_val),
                         textcoords='offset points',
                         xytext=(0, 10),
                         ha='center')

    if tbi_slo is not None:
        plt.axhline(y=tbi_slo, color='red', linestyle='--', label='TBI SLO')

    plt.xlabel('Images Per Second')
    plt.ylabel('Time Between Images')
    plt.title(plot_name)
    plt.grid(True)
    legend = plt.legend()
    legend._legend_box.sep = 3
    legend._legend_box.align = "right"
    plt.setp(legend.get_texts(), fontsize='small')
    plt.setp(legend.get_patches(), scalex=0.5, scaley=0.5)
    plt.tight_layout()
    plt.savefig('plotting/plots/' + plot_filename)
    

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
        gpu_type = curr_bmark_params[2]
        bmark_info = gpu_batch_exp_utils.parse_bmark_output(bmark_output_paths[i])
        nvsmi_info = gpu_batch_exp_utils.parse_nvsmi_output(nvsmi_output_paths[i])
        bmark_entry['model_size'] = model_size
        bmark_entry['batch_size'] = batch_size
        bmark_entry['gpu_type'] = gpu_type
        bmark_entry['bmark_info'] = bmark_info
        bmark_entry['nvsmi_info'] = nvsmi_info
        bmark_entries.append(bmark_entry)

    if args.plot_ips_vs_tbi:
        plot_ips_vs_tbi(
            bmark_entries,
            args.bmark_param_groups,
            args.plot_filename,
            args.plot_name,
            args.tbi_slo
        )
    if args.plot_tbi_vs_epi:
        plot_tbi_vs_epi(
            bmark_entries,
            args.bmark_param_groups,
            args.gpu_idxs,
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
        help='[model size] [batch size] [gpu type]'
    )
    parser.add_argument(
        '--bmark_param_groups',
        type=str,
        nargs='+',
        help='[model size] [batch size] [gpu type] (specify "X" for any value)'
    )
    parser.add_argument(
        '--plot_ips_vs_tbi',
        default=False,
        action='store_true',
        help='specify this arg to plot ips (images per second) vs tbi (time between images)'
    )
    parser.add_argument(
        '--plot_tbi_vs_epi',
        default=False,
        action='store_true',
        help='specify this arg to plot epi (energy per image) vs tbi (time between images)'
    )
    parser.add_argument(
        '--tbi_slo',
        type=float,
        help='specify the tbi SLO for human readability'
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
        '--gpu_idxs',
        nargs='+',
        type=int,
        help='specify which idx of GPU for nvsmi info'
    )
    args = parser.parse_args()
    main(args)
