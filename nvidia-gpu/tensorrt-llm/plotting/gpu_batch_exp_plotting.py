import argparse
import re
import ast
import matplotlib.pyplot as plt
import numpy as np
import math
from pathlib import Path

import gpu_batch_exp_utils
import carbon_data


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
    max_sequence_length = bmark_entry['max_sequence_length']
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
        if (plotting_knob != 'max_sequence_length' and
            bmark_param_group_dict['max_sequence_length'] != max_sequence_length):
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


# REPORTED COUNTS
def calculate_total_batch_generated_tokens(
    batch_dict,
    batch_size
):
    batch_input_lengths_sum, batch_output_lengths_sum = 0, 0
    batch_input_lengths_items = batch_dict['batch_input_lengths'].items()
    batch_output_lengths_items = batch_dict['batch_output_lengths'].items()
    assert(len(batch_input_lengths_items) == len(batch_output_lengths_items) and
           len(batch_output_lengths_items) == batch_size)

    for (batch_input_lengths_index, batch_input_lengths), (batch_output_lengths_index, batch_output_lengths) in zip(batch_input_lengths_items, batch_output_lengths_items):
        batch_input_lengths_sum += batch_input_lengths
        batch_output_lengths_sum += batch_output_lengths

    total_batch_generated_tokens = batch_output_lengths_sum - batch_input_lengths_sum
    return total_batch_generated_tokens


# Calcaulates TBT (Time Between Tokens) for different bmark data points and adds information to plotting dicts
def calculate_avg_tbt(
    bmark_entries,
    bmark_param_groups,
    plotting_knob,
    bmark_param_group_dicts
):
    for bmark_entry in bmark_entries:
        batch_size = bmark_entry['batch_size']
        bmark_info = bmark_entry['bmark_info']

        # For this given bmark data point, keep track of running tbt sum to calculate avg at the end
        tbt_sum = 0
        # Manually track the number of parsed iterations for edge cases
        num_iterations = 0

        for batch_iteration, batch_dict in bmark_info.items():
            batch_start_time = batch_dict['batch_start_time']
            batch_end_time = batch_dict['batch_end_time']
            batch_e2e_time = batch_end_time - batch_start_time

            total_batch_generated_tokens = calculate_total_batch_generated_tokens(
                batch_dict,
                batch_size
            )
            # if no tokens generated, skip this iteration
            if total_batch_generated_tokens == 0:
                continue

            avg_batch_generated_tokens = total_batch_generated_tokens / batch_size
            batch_tbt_avg = batch_e2e_time / avg_batch_generated_tokens
            tbt_sum += batch_tbt_avg
            num_iterations += 1

        avg_tbt = tbt_sum / num_iterations
        update_experiment_data(
            bmark_param_group_dicts,
            plotting_knob,
            'avg_tbt',
            avg_tbt,
            bmark_entry
        )


# Calcaulates TPS (Tokens Per Second) for different bmark data points and adds information to plotting dicts
def calculate_avg_tps(
    bmark_entries,
    bmark_param_groups,
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

            total_batch_generated_tokens = calculate_total_batch_generated_tokens(
                batch_dict,
                batch_size
            )
            # if no tokens generated, skip this iteration
            if total_batch_generated_tokens == 0:
                continue

            # tps = tokens per second
            batch_tps_avg = total_batch_generated_tokens / batch_e2e_time
            tps_sum += batch_tps_avg
            num_iterations += 1

        avg_tps = tps_sum / num_iterations
        update_experiment_data(
            bmark_param_group_dicts,
            plotting_knob,
            'avg_tps',
            avg_tps,
            bmark_entry
        )


def calculate_avg_ept(
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
        batch_total_tokens_list = []
        for batch_iteration, batch_dict in bmark_info.items():
            batch_start_times.append(batch_dict['batch_start_time'])
            batch_end_times.append(batch_dict['batch_end_time'])

            # Calculate and record the total tokens decoded during this batch
            total_batch_generated_tokens = calculate_total_batch_generated_tokens(
                batch_dict,
                batch_size
            )
            batch_total_tokens_list.append(total_batch_generated_tokens)

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
            #gpu_idx_dict = nvsmi_dict[gpu_idx]
            #nvsmi_curr_powers.append(gpu_idx_dict['curr_power_usage'])

        # For this given bmark data point, keep track of running tbt sum to calculate avg at the end
        ept_sum = 0
        # Manually track the number of parsed iterations for edge cases
        num_iterations = 0

        # For each bmark_entry, calculate the average energy per token across the bmark
        # Average EPT across every iteration in the bmark
        for i in range(len(batch_start_times)):
            batch_start_time, batch_end_time = batch_start_times[i], batch_end_times[i]
            total_batch_generated_tokens = batch_total_tokens_list[i]

            if total_batch_generated_tokens == 0:
                continue

            # Find the nvsmi timestamps and power metrics that correspond to this batch
            # - First nvsmi timestamp before batch starts
            # - First nvsmi timestamp after batch ends
            nvsmi_before_ts, nvsmi_after_ts = -1, -1
            for j in range(len(nvsmi_timestamps) - 1):
                nvsmi_timestamp_0, nvsmi_timestamp_1 = nvsmi_timestamps[j], nvsmi_timestamps[j+1]
                #nvsmi_curr_power_0, nvsmi_curr_power_1 = nvsmi_curr_powers[j], nvsmi_curr_powers[j+1]

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

            #energy_joules = np.trapz(batch_nvsmi_curr_powers, batch_nvsmi_timestamps)
            energy_joules_sum = 0
            for j in range(len(gpu_idxs)):
                energy_joules = np.trapz(batch_nvsmi_curr_powers_split[j], batch_nvsmi_timestamps)
                energy_joules_sum += energy_joules

            batch_ept_avg = energy_joules_sum / total_batch_generated_tokens
            ept_sum += batch_ept_avg
            num_iterations += 1

        avg_ept = ept_sum / num_iterations
        update_experiment_data(
            bmark_param_group_dicts,
            plotting_knob,
            'avg_ept',
            avg_ept,
            bmark_entry
        )


def joules_to_kWh(joules):
    kWh = joules / 3600000
    return kWh

def years_to_sec(years):
    sec = years * 365 * 24 * 60 * 60
    return sec

def sec_to_years(sec):
    years = sec / (365 * 24 * 60 * 60)
    return years

def g_to_kg(g):
    kg = g / 1000
    return kg


def plot_tco_breakeven(
    bmark_entries,
    bmark_param_groups,
    gpu_idxs,
    required_tps,        # the current load
    workload_duration_s, # how long are we running this load for? (in seconds)
    usd_per_kWh,         # USD per kWh (regional electricity price)
    PUE,                 # Power Usage Efficiency
    gpu_lifetime_y,      # expected lifetime of a GPU (in years)
    usd_per_a10040gb,
    usd_per_v10032gb,
    plot_filename,
    plot_name
):
    plotting_metrics = [
        'batch_size',
        'avg_ept',
        'avg_tps'
    ]
    bmark_param_group_dicts = group_experiment_data(
        bmark_entries,
        bmark_param_groups,
        plotting_metrics
    )
    plotting_knob = 'batch_size'
    for bmark_entry in bmark_entries:
        batch_size = bmark_entry['batch_size']
        update_experiment_data(
            bmark_param_group_dicts,
            plotting_knob,
            plotting_knob,
            batch_size,
            bmark_entry
        )

    # Calculate TPS
    calculate_avg_tps(
        bmark_entries,
        bmark_param_groups,
        plotting_knob,
        bmark_param_group_dicts
    )
    # Calculate EPT
    calculate_avg_ept(
        bmark_entries,
        bmark_param_groups,
        plotting_knob,
        gpu_idxs,
        bmark_param_group_dicts
    )

    bar_labels = []
    new_total_opex_costs = []
    new_total_capex_costs = []
    new_total_overall_costs = []
    breakeven_lifetimes = []
    breakeven_groups = {}
    # Group together data from the same model but different GPU
    for bmark_param_group_dict in bmark_param_group_dicts:
        print('\n\n')
        for key, val in bmark_param_group_dict.items():
            print(f'{key}: {val}')

        if bmark_param_group_dict['model_size'] not in breakeven_groups:
            breakeven_groups[bmark_param_group_dict['model_size']] = {}
        breakeven_groups[bmark_param_group_dict['model_size']][bmark_param_group_dict['gpu_type']] = bmark_param_group_dict

    for model_size, gpu_entries in breakeven_groups.items():
        # First, calculate costs for a10040gb (new)
        new_bmark_param_group_dict = gpu_entries['a10040gb']
        gpu_price = usd_per_a10040gb
        new_avg_tps = new_bmark_param_group_dict['avg_tps']
        new_avg_ept = new_bmark_param_group_dict['avg_ept']
        new_batch_size = new_bmark_param_group_dict['batch_size']
        assert(len(new_avg_tps) == len(new_avg_ept) and
               len(new_avg_ept) == len(new_batch_size))

        for avg_tps_val, avg_ept_val, batch_size_val in zip(new_avg_tps, new_avg_ept, new_batch_size):
            # Calculate the number of required GPUs
            # (tokens / sec) / ((tokens / sec) / gpu)
            num_gpus_req = math.ceil(required_tps / avg_tps_val)

            # Calculate the total energy required to compute the workload
            # (joules / token) * (tokens / sec) * sec
            total_energy_joules = avg_ept_val * required_tps * workload_duration_s * PUE
            total_energy_kWh = joules_to_kWh(total_energy_joules)

            # Calculate OpEx costs from energy usage and rate
            total_opex_cost = total_energy_kWh * usd_per_kWh

            # Calculate CapEx costs from workload duration, single gpu price, and gpu lifetime
            gpu_lifetime_s = years_to_sec(gpu_lifetime_y)
            total_capex_cost = num_gpus_req * gpu_price * (workload_duration_s / gpu_lifetime_s)
            total_overall_cost = total_opex_cost + total_capex_cost

            model_size = bmark_param_group_dict['model_size']
            gpu_type = bmark_param_group_dict['gpu_type']
            bar_labels.append(f'{model_size}_{gpu_type}_{batch_size_val}')
            new_total_opex_costs.append(total_opex_cost)
            new_total_capex_costs.append(total_capex_cost)
            new_total_overall_costs.append(total_overall_cost)

        # Then, calculate breakeven for v10032gb (old) given a10040gb cost
        old_bmark_param_group_dict = gpu_entries['v10032gb']
        gpu_price = usd_per_v10032gb
        old_avg_tps = old_bmark_param_group_dict['avg_tps']
        old_avg_ept = old_bmark_param_group_dict['avg_ept']
        old_batch_size = old_bmark_param_group_dict['batch_size']
        print(f'old_avg_tps: {old_avg_tps}')
        print(f'old_avg_ept: {old_avg_ept}')
        print(f'old_batch_size: {old_batch_size}')
        print(f'new_total_overall_costs: {new_total_overall_costs}')
        assert(len(old_avg_tps) == len(old_avg_ept) and
               len(old_avg_ept) == len(old_batch_size) and
               len(old_batch_size) == len(new_total_overall_costs))

        for avg_tps_val, avg_ept_val, batch_size_val, new_total_overall_cost_val in zip(old_avg_tps, old_avg_ept, old_batch_size, new_total_overall_costs):
            # Find how many years it takes
            workload_joules_per_second = avg_ept_val * required_tps * PUE
            workload_kWh_per_second = joules_to_kWh(workload_joules_per_second)
            workload_usd_per_second = workload_kWh_per_second * usd_per_kWh

            breakeven_lifetime_s = new_total_overall_cost_val / workload_usd_per_second
            breakeven_lifetime_y = sec_to_years(breakeven_lifetime_s)
            breakeven_lifetimes.append(breakeven_lifetime_y)

    print(f'new_total_overall_costs: {new_total_overall_costs}')
    print(f'breakeven_lifetimes: {breakeven_lifetimes}')

    #fig, ax = plt.subplots()
    #bar_width = 0.5
    #bar_positions = range(len(bar_labels))
    #capex_bars = ax.bar(bar_positions, total_capex_costs, bar_width, label='CapEx')
    #opex_bars = ax.bar(bar_positions, total_opex_costs, bar_width, bottom=total_capex_costs, label='OpEx')
    #ax.set_xlabel('Workload')
    #ax.set_ylabel('Cost (USD)')
    #ax.set_title(plot_name)
    #ax.set_xticks(bar_positions)
    #ax.set_xticklabels(bar_labels, rotation=45, ha='right', fontsize=8)
    #ax.legend()

    ## Add cost number to capex and opex bars
    #for bar, capex, opex in zip(capex_bars, total_capex_costs, total_opex_costs):
    #    height = bar.get_height()
    #    ax.text(
    #        bar.get_x() + bar.get_width() / 2.0, height / 2,
    #        f'{capex:.1f}', ha='center', va='bottom', color='black', fontsize=8
    #    )
    #    ax.text(
    #        bar.get_x() + bar.get_width() / 2.0, height + opex / 2,
    #        f'{opex:.1f}', ha='center', va='bottom', color='black', fontsize=8
    #    )

    #plt.tight_layout(pad=2.0)
    #plt.savefig('plots/' + plot_filename)


# Currently just plots on-premise GPU situation
# TODO: Add cloud carbon calculation
# TODO: This code is exactly the same as plot_tco_breakdown... basically just different names for cost variables
def plot_tcf_breakdown(
    bmark_entries,
    bmark_param_groups,
    gpu_idxs,
    required_tps,        # the current load
    workload_duration_s, # how long are we running this load for? (in seconds)
    gCO2eq_per_kWh,      # gC02eq per kWh (regional carbon intensity)
    PUE,                 # Power Usage Efficiency
    gpu_lifetime_y,      # expected lifetime of a GPU (in years)
    kgCO2eq_per_a10040gb,
    kgCO2eq_per_v10032gb,
    plot_filename,
    plot_name
):
    plotting_metrics = [
        'batch_size',
        'avg_ept',
        'avg_tps'
    ]
    bmark_param_group_dicts = group_experiment_data(
        bmark_entries,
        bmark_param_groups,
        plotting_metrics
    )
    plotting_knob = 'batch_size'
    for bmark_entry in bmark_entries:
        batch_size = bmark_entry['batch_size']
        update_experiment_data(
            bmark_param_group_dicts,
            plotting_knob,
            plotting_knob,
            batch_size,
            bmark_entry
        )

    # Calculate TPS
    calculate_avg_tps(
        bmark_entries,
        bmark_param_groups,
        plotting_knob,
        bmark_param_group_dicts
    )
    # Calculate EPT
    calculate_avg_ept(
        bmark_entries,
        bmark_param_groups,
        plotting_knob,
        gpu_idxs,
        bmark_param_group_dicts
    )

    bar_labels = []
    total_operational_carbons = []
    total_embodied_carbons = []
    total_overall_carbons = []
    # 1) Find out what is the <avg_tps_max> (and corresponding <avg_ept_max>) for each GPU/model configuration
    # 2) Find out the minimum number of this GPU is required to serve the required_tps load
    # 3) Calculate total energy required to compute <required_tps> for <workload_duration_s> given <avg_ept>
    for bmark_param_group_dict in bmark_param_group_dicts:
        print('\n\n')
        for key, val in bmark_param_group_dict.items():
            print(f'{key}: {val}')

        if bmark_param_group_dict['gpu_type'] == 'a10040gb':
            gpu_embodied = kgCO2eq_per_a10040gb
        elif bmark_param_group_dict['gpu_type'] == 'v10032gb':
            gpu_embodied = kgCO2eq_per_v10032gb
        else:
            raise ValueError('plot_tcf_breakdown: gpu_type not found')

        avg_tps = bmark_param_group_dict['avg_tps']
        avg_ept = bmark_param_group_dict['avg_ept']
        batch_size = bmark_param_group_dict['batch_size']
        assert(len(avg_tps) == len(avg_ept) and
               len(avg_ept) == len(batch_size))

        for avg_tps_val, avg_ept_val, batch_size_val in zip(avg_tps, avg_ept, batch_size):
            # Calculate the number of required GPUs
            # (tokens / sec) / ((tokens / sec) / gpu)
            num_gpus_req = math.ceil(required_tps / avg_tps_val)

            # Calculate the total energy required to compute the workload
            # (joules / token) * (tokens / sec) * sec
            total_energy_joules = avg_ept_val * required_tps * workload_duration_s * PUE
            total_energy_kWh = joules_to_kWh(total_energy_joules)

            # Calculate operational carbon from energy usage and carbon intensity
            total_operational_carbon_g = total_energy_kWh * gCO2eq_per_kWh
            total_operational_carbon_kg = g_to_kg(total_operational_carbon_g)

            # Calculate embodied_carbon (kgCO2eq) from workload duration, single gpu embodied, and gpu lifetime
            gpu_lifetime_s = years_to_sec(gpu_lifetime_y)
            total_embodied_carbon = num_gpus_req * gpu_embodied * (workload_duration_s / gpu_lifetime_s)
            total_overall_carbon = total_operational_carbon_kg + total_embodied_carbon

            model_size = bmark_param_group_dict['model_size']
            gpu_type = bmark_param_group_dict['gpu_type']
            bar_labels.append(f'{model_size}_{gpu_type}_{batch_size_val}')
            total_operational_carbons.append(total_operational_carbon_kg)
            total_embodied_carbons.append(total_embodied_carbon)
            total_overall_carbons.append(total_overall_carbon)

    fig, ax = plt.subplots()
    bar_width = 0.5
    bar_positions = range(len(bar_labels))
    embodied_bars = ax.bar(bar_positions, total_embodied_carbons, bar_width, label='Embodied')
    operational_bars = ax.bar(bar_positions, total_operational_carbons, bar_width, bottom=total_embodied_carbons, label='Operational')
    ax.set_xlabel('Workload')
    ax.set_ylabel('Carbon (kgCO2eq)')
    ax.set_title(plot_name)
    ax.set_xticks(bar_positions)
    ax.set_xticklabels(bar_labels, rotation=45, ha='right', fontsize=8)
    ax.legend()

    # Add cost number to capex and opex bars
    for bar, embodied_carbon, operational_carbon in zip(embodied_bars, total_embodied_carbons, total_operational_carbons):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0, height / 2,
            f'{embodied_carbon:.1f}', ha='center', va='bottom', color='black', fontsize=8
        )
        ax.text(
            bar.get_x() + bar.get_width() / 2.0, height + operational_carbon / 2,
            f'{operational_carbon:.1f}', ha='center', va='bottom', color='black', fontsize=8
        )

    plt.tight_layout(pad=2.0)
    plt.savefig('plots/' + plot_filename)


# Currently just plots on-premise GPU situation
# TODO: Add cloud cost calculation
def plot_tco_breakdown(
    bmark_entries,
    bmark_param_groups,
    gpu_idxs,
    required_tps,        # the current load
    workload_duration_s, # how long are we running this load for? (in seconds)
    usd_per_kWh,         # USD per kWh (regional electricity price)
    PUE,                 # Power Usage Efficiency
    gpu_lifetime_y,      # expected lifetime of a GPU (in years)
    usd_per_a10040gb,
    usd_per_v10032gb,
    plot_filename,
    plot_name
):
    plotting_metrics = [
        'batch_size',
        'avg_ept',
        'avg_tps'
    ]
    bmark_param_group_dicts = group_experiment_data(
        bmark_entries,
        bmark_param_groups,
        plotting_metrics
    )
    plotting_knob = 'batch_size'
    for bmark_entry in bmark_entries:
        batch_size = bmark_entry['batch_size']
        update_experiment_data(
            bmark_param_group_dicts,
            plotting_knob,
            plotting_knob,
            batch_size,
            bmark_entry
        )

    # Calculate TPS
    calculate_avg_tps(
        bmark_entries,
        bmark_param_groups,
        plotting_knob,
        bmark_param_group_dicts
    )
    # Calculate EPT
    calculate_avg_ept(
        bmark_entries,
        bmark_param_groups,
        plotting_knob,
        gpu_idxs,
        bmark_param_group_dicts
    )

    bar_labels = []
    total_opex_costs = []
    total_capex_costs = []
    total_overall_costs = []
    # 1) Find out what is the <avg_tps_max> (and corresponding <avg_ept_max>) for each GPU/model configuration
    # 2) Find out the minimum number of this GPU is required to serve the required_tps load
    # 3) Calculate total energy required to compute <required_tps> for <workload_duration_s> given <avg_ept>
    for bmark_param_group_dict in bmark_param_group_dicts:
        for key, val in bmark_param_group_dict.items():
            print(f'{key}: {val}')

        if bmark_param_group_dict['gpu_type'] == 'a10040gb':
            gpu_price = usd_per_a10040gb
        elif bmark_param_group_dict['gpu_type'] == 'v10032gb':
            gpu_price = usd_per_v10032gb
        else:
            raise ValueError('plot_tco_breakdown: gpu_type not found')

        avg_tps = bmark_param_group_dict['avg_tps']
        avg_ept = bmark_param_group_dict['avg_ept']
        batch_size = bmark_param_group_dict['batch_size']
        assert(len(avg_tps) == len(avg_ept) and
               len(avg_ept) == len(batch_size))

        for avg_tps_val, avg_ept_val, batch_size_val in zip(avg_tps, avg_ept, batch_size):
            # Calculate the number of required GPUs
            # (tokens / sec) / ((tokens / sec) / gpu)
            num_gpus_req = math.ceil(required_tps / avg_tps_val)

            # Calculate the total energy required to compute the workload
            # (joules / token) * (tokens / sec) * sec
            total_energy_joules = avg_ept_val * required_tps * workload_duration_s * PUE
            total_energy_kWh = joules_to_kWh(total_energy_joules)

            # Calculate OpEx costs from energy usage and rate
            total_opex_cost = total_energy_kWh * usd_per_kWh

            # Calculate CapEx costs from workload duration, single gpu price, and gpu lifetime
            gpu_lifetime_s = years_to_sec(gpu_lifetime_y)
            total_capex_cost = num_gpus_req * gpu_price * (workload_duration_s / gpu_lifetime_s)
            total_overall_cost = total_opex_cost + total_capex_cost

            model_size = bmark_param_group_dict['model_size']
            gpu_type = bmark_param_group_dict['gpu_type']
            bar_labels.append(f'{model_size}_{gpu_type}_{batch_size_val}')
            total_opex_costs.append(total_opex_cost)
            total_capex_costs.append(total_capex_cost)
            total_overall_costs.append(total_overall_cost)

    fig, ax = plt.subplots()
    bar_width = 0.5
    bar_positions = range(len(bar_labels))
    capex_bars = ax.bar(bar_positions, total_capex_costs, bar_width, label='CapEx')
    opex_bars = ax.bar(bar_positions, total_opex_costs, bar_width, bottom=total_capex_costs, label='OpEx')
    ax.set_xlabel('Workload')
    ax.set_ylabel('Cost (USD)')
    ax.set_title(plot_name)
    ax.set_xticks(bar_positions)
    ax.set_xticklabels(bar_labels, rotation=45, ha='right', fontsize=8)
    ax.legend()

    # Add cost number to capex and opex bars
    for bar, capex, opex in zip(capex_bars, total_capex_costs, total_opex_costs):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0, height / 2,
            f'{capex:.1f}', ha='center', va='bottom', color='black', fontsize=8
        )
        ax.text(
            bar.get_x() + bar.get_width() / 2.0, height + opex / 2,
            f'{opex:.1f}', ha='center', va='bottom', color='black', fontsize=8
        )

    plt.tight_layout(pad=2.0)
    plt.savefig('plots/' + plot_filename)


def plot_tbt_vs_ept(
    bmark_entries,
    bmark_param_groups,
    gpu_idxs,
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
        batch_size = bmark_entry['batch_size']
        update_experiment_data(
            bmark_param_group_dicts,
            plotting_knob,
            plotting_knob,
            batch_size,
            bmark_entry
        )

    # Calculate TBT
    calculate_avg_tbt(
        bmark_entries,
        bmark_param_groups,
        plotting_knob,
        bmark_param_group_dicts
    )
    # Calculate EPT
    calculate_avg_ept(
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

        avg_ept = bmark_param_group_dict['avg_ept']
        avg_tbt = bmark_param_group_dict['avg_tbt']
        batch_sizes = bmark_param_group_dict['batch_size']

        model_size = bmark_param_group_dict['model_size']
        max_sequence_length = bmark_param_group_dict['max_sequence_length']
        gpu_type = bmark_param_group_dict['gpu_type']
        plt.plot(avg_tbt, avg_ept, label=f'{model_size} {gpu_type}', marker='o')

        for avg_tbt_val, avg_ept_val, batch_size in zip(avg_tbt, avg_ept, batch_sizes):
            plt.annotate(str(batch_size),
                         (avg_tbt_val, avg_ept_val),
                         textcoords='offset points',
                         xytext=(0, 10),
                         ha='center')

    plt.xlabel('Time Between Tokens')
    plt.ylabel('Joules Per Token')
    plt.title(plot_name)
    plt.grid(True)
    legend = plt.legend()
    legend._legend_box.sep = 3
    legend._legend_box.align = "right"
    plt.setp(legend.get_texts(), fontsize='small')
    plt.setp(legend.get_patches(), scalex=0.5, scaley=0.5)
    plt.tight_layout()
    plt.savefig('plots/' + plot_filename)


# NOTE
#  - This is theoretical user-perceived latency to provide a bound for tbt (time between tokens).
#  - Actual user-perceived latency depends on how quickly the new tokens actually make it to the user.
#  - TTFT (time to first token) is also an important metric, but is not taken into account with these experiments.
# throughput : tokens per second
# latency    : theoretical user-perceived seconds per token (tbt)
def plot_tps_vs_tbt(
    bmark_entries,
    bmark_param_groups,
    plot_filename,
    plot_name,
    add_human_slo
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
        batch_size = bmark_entry['batch_size']
        update_experiment_data(
            bmark_param_group_dicts,
            plotting_knob,
            plotting_knob,
            batch_size,
            bmark_entry
        )

    # Calculate TBT
    calculate_avg_tbt(
        bmark_entries,
        bmark_param_groups,
        plotting_knob,
        bmark_param_group_dicts
    )
    # Calculate TPS
    calculate_avg_tps(
        bmark_entries,
        bmark_param_groups,
        plotting_knob,
        bmark_param_group_dicts
    )

    plt.figure(figsize=(8, 3))
    for bmark_param_group_dict in bmark_param_group_dicts:
        for key, val in bmark_param_group_dict.items():
            print(f'{key}: {val}')

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
    plt.ylabel('Time Between Tokens')
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
#def plot_normalized_token_latency(
#    bmark_entries,
#    plot_filename,
#    bmark_param_groups,
#    excluded_tokens
#):
#    plt.figure(figsize=(10, 5))
#    bmark_param_group_dicts = []
#    for bmark_param_group in bmark_param_groups:
#        group_split = bmark_param_group.split()
#        bmark_param_group_dict = {}
#        bmark_param_group_dict['model_size'] = int(group_split[0]) if group_split[0] != 'X' else 'X'
#        bmark_param_group_dict['batch_size'] = int(group_split[1]) if group_split[1] != 'X' else 'X'
#        bmark_param_group_dict['max_sequence_length'] = int(group_split[2]) if group_split[2] != 'X' else 'X'
#        bmark_param_group_dict['gpu_type'] = group_split[3] if group_split[3] != 'X' else 'X'
#        bmark_param_group_dict['batch_sweep_info'] = []
#        bmark_param_group_dicts.append(bmark_param_group_dict)
#
#    for bmark_entry in bmark_entries:
#        model_size = bmark_entry['model_size']
#        batch_size = bmark_entry['batch_size']
#        max_sequence_length = bmark_entry['max_sequence_length']
#        gpu_type = bmark_entry['gpu_type']
#        batch_sweep_info = {
#            'model_size': model_size,
#            'batch_size': batch_size,
#            'max_sequence_length': max_sequence_length,
#            'gpu_type': gpu_type
#        }
#        print(f'bmark_entry: {model_size} {batch_size} {max_sequence_length} {gpu_type}')
#
#        # Extract timestamps from bmark_info
#        bmark_info = bmark_entry['bmark_info']
#        # keeping running sum of normalized token latencies to average at the end for this bmark
#        normalized_token_latency_sum = 0
#        included_normalized_token_latency_sum = 0
#        e2e_batch_latency_sum = 0
#        total_batch_output_lengths_sum = 0
#        total_included_batch_output_lengths_sum = 0
#
#        # each entry is (batch_start_time, batch_end_time)
#        curr_max_time = 0.0
#        num_iterations = 0
#        for batch_iteration, batch_dict in bmark_info.items():
#            num_iterations += 1
#            batch_start_time = batch_dict['batch_start_time']
#            batch_end_time = batch_dict['batch_end_time']
#
#            # make sure timestamps are strictly increasing
#            assert(batch_start_time > curr_max_time and
#                   batch_end_time > batch_start_time)
#            curr_max_time = batch_end_time
#            # TODO: w/o continuous batching, latency of every request in batch is the same
#            e2e_batch_latency = batch_end_time - batch_start_time
#            e2e_batch_latency_sum += e2e_batch_latency
#
#            batch_size = batch_dict['batch_size']
#            total_batch_output_lengths = 0
#            total_included_batch_output_lengths = 0
#            for i in range(batch_size):
#                batch_input_tokens = batch_dict['batch_input_tokens'][i]
#                batch_input_lengths = batch_dict['batch_input_lengths'][i]
#                batch_output_tokens = batch_dict['batch_output_tokens'][i]
#                batch_output_lengths = batch_dict['batch_output_lengths'][i]
#
#                # count non-padding tokens (or any excluded tokens)
#                included_batch_input_lengths, included_batch_output_lengths = 0, 0
#                excluded_batch_input_lengths, excluded_batch_output_lengths = 0, 0
#                included_batch_input_tokens, included_batch_output_tokens = [], []
#                excluded_batch_input_tokens, excluded_batch_output_tokens = [], []
#                for token_id in batch_input_tokens: # TODO: these values currently are not used anywhere
#                    assert(type(token_id) == int)
#                    if token_id in excluded_tokens:
#                        excluded_batch_input_tokens.append(token_id)
#                        excluded_batch_input_lengths += 1
#                    else:
#                        included_batch_input_tokens.append(token_id)
#                        included_batch_input_lengths += 1
#                for token_id in batch_output_tokens:
#                    assert(type(token_id) == int)
#                    if token_id in excluded_tokens:
#                        excluded_batch_output_tokens.append(token_id)
#                        excluded_batch_output_lengths += 1
#                    else:
#                        included_batch_output_tokens.append(token_id)
#                        included_batch_output_lengths += 1
#
#                # add to token sums for this batch
#                total_batch_output_lengths += batch_output_lengths
#                total_included_batch_output_lengths += included_batch_output_lengths
#
#                # verify lengths for this batch
#                assert(len(batch_input_tokens) == batch_input_lengths and
#                       len(batch_output_tokens) == (included_batch_output_lengths + excluded_batch_output_lengths))
#
#            # calculate normalized token latencies for this current batch
#            batch_normalized_token_latency = e2e_batch_latency / total_batch_output_lengths
#            included_batch_normalized_token_latency = e2e_batch_latency / total_included_batch_output_lengths
#            total_batch_output_lengths_sum += total_batch_output_lengths
#            total_included_batch_output_lengths_sum += total_included_batch_output_lengths
#
#            # add latencies to running sums
#            normalized_token_latency_sum += batch_normalized_token_latency
#            included_normalized_token_latency_sum += included_batch_normalized_token_latency
#
#        # calculate actual normalized token latencies for entire bmark
#        batch_sweep_info['normalized_token_latency'] = normalized_token_latency_sum / num_iterations
#        batch_sweep_info['included_normalized_token_latency'] = included_normalized_token_latency_sum / num_iterations
#        batch_sweep_info['avg_e2e_batch_latency'] = e2e_batch_latency_sum / num_iterations
#        batch_sweep_info['avg_output_tokens_per_batch'] = total_batch_output_lengths_sum / num_iterations
#        batch_sweep_info['avg_included_output_tokens_per_batch'] = total_included_batch_output_lengths_sum / num_iterations
#        #batch_sweep_infos.append(batch_sweep_info)
#
#        # calculate avg FLOPs per batch 7B
#        avg_flops_per_batch = transformer_model_scaling.calculate_transformer_flops(
#            4096,
#            32,
#            4096,
#            4096,
#            int(batch_sweep_info['avg_output_tokens_per_batch'])
#        )
#        batch_sweep_info['avg_flops_per_batch'] = avg_flops_per_batch
#        batch_sweep_info['avg_tflops_per_batch'] = avg_flops_per_batch / (10 ** 12)
#
#        # calculate TFLOPs achievable in avg batch time by a100 (624 FP16 TFLOPS)
#        #batch_sweep_info['peak_TFLOPs_in_batch_time'] = batch_sweep_info['avg_e2e_batch_latency'] * 624
#        #batch_sweep_info['avg_GPU_utilization'] = batch_sweep_info['avg_tflops_per_batch'] / batch_sweep_info['peak_TFLOPs_in_batch_time']
#
#        # TFLOPs achievable v100 (130 TFLOPS)
#        batch_sweep_info['peak_TFLOPs_in_batch_time'] = batch_sweep_info['avg_e2e_batch_latency'] * 130
#        batch_sweep_info['avg_GPU_utilization'] = batch_sweep_info['avg_tflops_per_batch'] / batch_sweep_info['peak_TFLOPs_in_batch_time']
#
#        # group things in to their bmark param group
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
#            if (bmark_param_group_dict['gpu_type'] != 'X' and
#                bmark_param_group_dict['gpu_type'] != gpu_type):
#                continue
#
#            # Only reach this point if a match is found
#            bmark_param_match_found = True
#            bmark_param_group_dict['batch_sweep_info'].append(batch_sweep_info)
#            break
#
#        # For each bmark_entry, should at least match to one of the plotting groups
#        assert(bmark_param_match_found)
#
#    plot_batch_sizes = []
#    plot_gpu_utilization = []
#    for bmark_param_group_dict in bmark_param_group_dicts:
#        #for key, value in bmark_param_group_dict.items():
#        #    print(f'{key}: {value}')
#        batch_sweep_info = bmark_param_group_dict['batch_sweep_info']
#        for info_dict in batch_sweep_info:
#            print(f'{info_dict["batch_size"]}: {info_dict["avg_GPU_utilization"]}')
#            plot_batch_sizes.append(info_dict['batch_size'])
#            plot_gpu_utilization.append(info_dict['avg_GPU_utilization'] * 100)
#
#    plt.figure(figsize=(10, 5))
#    #plt.bar(plot_batch_sizes, plot_gpu_utilization, color='blue')
#    plt.plot(plot_batch_sizes, plot_gpu_utilization, marker='o', linestyle='-', color='blue')
#
#    plt.xlabel('Batch Size')
#    plt.ylabel('GPU Utilization (%)')
#    #plt.title('A100 40GB SXM Utilization w/ Llama 7B')
#    plt.title('V100 32GB PCIE Utilization w/ Llama 7B')
#    plt.xticks(plot_batch_sizes)
#    plt.grid(True)
#    plt.savefig('plots/' + 'v10032gb_llama7b_utilization.png')

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
            args.plot_filename,
            args.plot_name,
            args.add_human_slo
        )
    if args.plot_tbt_vs_ept:
        plot_tbt_vs_ept(
            bmark_entries,
            args.bmark_param_groups,
            args.gpu_idxs,
            args.plot_filename,
            args.plot_name
        )
    if args.plot_tco_breakdown:
        plot_tco_breakdown(
            bmark_entries,
            args.bmark_param_groups,
            args.gpu_idxs,
            args.required_tps,
            args.workload_duration_s,
            args.usd_per_kWh,
            args.pue,
            args.gpu_lifetime_y,
            args.usd_per_a10040gb,
            args.usd_per_v10032gb,
            args.plot_filename,
            args.plot_name
        )
    if args.plot_tcf_breakdown:
        plot_tcf_breakdown(
            bmark_entries,
            args.bmark_param_groups,
            args.gpu_idxs,
            args.required_tps,
            args.workload_duration_s,
            args.gCO2eq_per_kWh,
            args.pue,
            args.gpu_lifetime_y,
            args.kgCO2eq_per_a10040gb,
            args.kgCO2eq_per_v10032gb,
            args.plot_filename,
            args.plot_name
        )
    if args.plot_tco_breakeven:
        plot_tco_breakeven(
            bmark_entries,
            args.bmark_param_groups,
            args.gpu_idxs,
            args.required_tps,
            args.workload_duration_s,
            args.usd_per_kWh,
            args.pue,
            args.gpu_lifetime_y,
            args.usd_per_a10040gb,
            args.usd_per_v10032gb,
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
        '--plot_tbt_vs_ept',
        default=False,
        action='store_true',
        help='specify this arg to plot ept (energy per token) vs tbt (time between tokens)'
    )
    parser.add_argument(
        '--plot_tco_breakdown',
        default=False,
        action='store_true',
        help='specify this arg to plot the tco breakdown (CapEx vs. OpEx) of inference serving'
    )
    parser.add_argument(
        '--plot_tcf_breakdown',
        default=False,
        action='store_true',
        help='specify this arg to plot the tcf breakdown (Embodied vs. Operational) of inference serving'
    )
    parser.add_argument(
        '--plot_tco_breakeven',
        default=False,
        action='store_true',
        help='specify this arg to plot the tco breakeven point of inference serving between old and new generation of GPUs'
    )
    parser.add_argument(
        '--add_human_slo',
        type=int,
        help='specify the TBT SLO for human readability'
    )
    parser.add_argument(
        '--required_tps',
        type=int,
        help='specify what token generation rate to simulate serving in tokens per second'
    )
    parser.add_argument(
        '--workload_duration_s',
        type=int,
        help='specify how long the simulated workload is running for in seconds'
    )
    parser.add_argument(
        '--usd_per_kWh',
        type=float,
        help='specify the regional electricity cost rate in usd per kWh'
    )
    parser.add_argument(
        '--gCO2eq_per_kWh',
        type=int,
        help='specify the regional carbon intensity in gCO2eq per kWh'
    )
    parser.add_argument(
        '--pue',
        type=float,
        help='specify the PUE (Power Usage Efficiency) of the simulated datacenter environment'
    )
    parser.add_argument(
        '--gpu_lifetime_y',
        type=int,
        help='specify the lifetime of datacenter GPU platforms in years'
    )
    parser.add_argument(
        '--usd_per_a10040gb',
        type=int,
        help='specify the price in USD of a single 40GB SXM4 A100 GPU'
    )
    parser.add_argument(
        '--usd_per_v10032gb',
        type=int,
        help='specify the price in USD of a single 32GB PCIE V100S GPU'
    )
    parser.add_argument(
        '--kgCO2eq_per_a10040gb',
        type=int,
        help='specify the estimated embodied carbon of manufacturing a single 40GB SXM4 A100 GPU'
    )
    parser.add_argument(
        '--kgCO2eq_per_v10032gb',
        type=int,
        help='specify the estimated embodied carbon of manufacturing a single 32GB PCIE V100S GPU'
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
