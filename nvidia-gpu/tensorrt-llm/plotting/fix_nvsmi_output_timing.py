import argparse
import re
import ast
import os
import shutil
import datetime

import gpu_batch_exp_utils


def correlate_bmark_nvsmi_timestamps(
    bmark_output_path: str,
    nvsmi_output_path: str,
    gpu_idx: int,
    nvsmi_start_line: int,
    nvsmi_end_line: int
):
    bmark_output = gpu_batch_exp_utils.parse_bmark_output(bmark_output_path)
    nvsmi_output = gpu_batch_exp_utils.parse_nvsmi_output(nvsmi_output_path)

    # extract the start and end timestamps from bmark output
    first_bmark_entry = bmark_output[0]
    last_bmark_entry = bmark_output[len(bmark_output)-1]
    bmark_start_time = first_bmark_entry['batch_start_time']
    bmark_end_time = last_bmark_entry['batch_end_time']
    bmark_e2e_time = bmark_end_time - bmark_start_time

    # nvsmi output file has an extra line on the top w/ the GPU name, and is also 1-indexed
    bmark_start_index = nvsmi_start_line - 2
    bmark_end_index = nvsmi_end_line - 2
    nvsmi_timing_interval = bmark_e2e_time / (bmark_end_index - bmark_start_index)

    #for nvsmi_dict in nvsmi_output:
    for i in range(len(nvsmi_output)):
        nvsmi_dict = nvsmi_output[i]

        # assume non-bmark nvsmi timing interval is 1 second
        if i < bmark_start_index:
            calculated_timestamp = bmark_start_time - (bmark_start_index - i)
        elif i > bmark_end_index:
            calculated_timestamp = bmark_end_time + (i - bmark_end_index)
        else:
            calculated_timestamp = bmark_start_time + ((i - bmark_start_index) * nvsmi_timing_interval)

        timestamp_readable = datetime.datetime.fromtimestamp(calculated_timestamp).strftime("%H:%M:%S")
        nvsmi_dict['timestamp_raw'] = calculated_timestamp
        nvsmi_dict['timestamp_readable'] = timestamp_readable

    return nvsmi_output


def main(args):
    bmark_output_path = args.bmark_output_path
    nvsmi_output_path = args.nvsmi_output_path
    gpu_idx = args.gpu_idx
    nvsmi_start_line = args.nvsmi_start_line
    nvsmi_end_line = args.nvsmi_end_line

    # save the old output file before modifying, but only if one doesn't already exist
    if not os.path.exists(bmark_output_path):
        save_output_path = bmark_output_path.replace('.out', '_old.out')
        shutil.copy(bmark_output_path, save_output_path)
    if not os.path.exists(nvsmi_output_path):
        save_output_path = nvsmi_output_path.replace('.out', '_old.out')
        shutil.copy(nvsmi_output_path, save_output_path)

    nvsmi_output = correlate_bmark_nvsmi_timestamps(
        bmark_output_path,
        nvsmi_output_path,
        gpu_idx,
        nvsmi_start_line,
        nvsmi_end_line
    )

    gpu_batch_exp_utils.write_nvsmi_output(nvsmi_output_path, nvsmi_output)


# This script adds timestamp information to nvsmi output files by correlating gpu utilization with the start and end of the batch experiments
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--bmark_output_path',
        type=str,
        required=True,
        help='paths to bmark output files'
    )
    parser.add_argument(
        '--nvsmi_output_path',
        type=str,
        required=True,
        help='paths to nvsmi output files'
    )
    parser.add_argument(
        '--gpu_idx',
        type=int,
        required=True,
        help='nvsmi gpu idx to use for reading utilization'
    )
    parser.add_argument(
        '--nvsmi_start_line',
        type=int,
        required=True,
        help='nvsmi line on which the bmark starts'
    )
    parser.add_argument(
        '--nvsmi_end_line',
        type=int,
        required=True,
        help='nvsmi line on which the bmark ends'
    )
    args = parser.parse_args()
    main(args)
