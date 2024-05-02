import argparse
import re
import ast
import os
import shutil

import gpu_batch_exp_utils


def correlate_bmark_nvsmi_timestamps(
    bmark_output_path: str,
    nvsmi_output_path: str
):
    bmark_output = gpu_batch_exp_utils.parse_bmark_output(bmark_output_path)
    nvsmi_output = gpu_batch_exp_utils.parse_nvsmi_output(nvsmi_output_path)


def main(args):
    bmark_output_paths = args.bmark_output_paths
    nvsmi_output_paths = args.nvsmi_output_paths
    gpu_idx = args.gpu_idx

    assert(len(bmark_output_paths) == len(nvsmi_output_paths))
    for i in range(len(bmark_output_paths)):
        bmark_output_path = bmark_output_paths[i]
        nvsmi_output_path = nvsmi_output_paths[i]

        # save the old output file before modifying, but only if one doesn't already exist
        if not os.path.exists(bmark_output_path):
            save_output_path = bmark_output_path.replace('.out', '_old.out')
            shutil.copy(bmark_output_path, save_output_path)
        if not os.path.exists(nvsmi_output_path):
            save_output_path = nvsmi_output_path.replace('.out', '_old.out')
            shutil.copy(nvsmi_output_path, save_output_path)

        correlate_bmark_nvsmi_timestamps(
            bmark_output_path,
            nvsmi_output_path
        )


# This script adds timestamp information to nvsmi output files by correlating gpu utilization with the start and end of the batch experiments
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
        '--gpu_idx',
        type=int,
        required=True,
        help='nvsmi gpu idx to use for reading utilization'
    )
    args = parser.parse_args()
    main(args)
