import argparse
import re
import ast
import os
import shutil


batch_output_lengths_pattern_old = r'batch_output_lengths\[(\d+)\]: \[(.*?)\]'
#batch_output_lengths_pattern_new = r'batch_output_lengths\[(\d+)\]'

def format_bmark_output(bmark_output_path: str):
    with open(bmark_output_path, 'r') as f:
        bmark_output_lines = f.readlines()

    modified_lines = []
    for line in bmark_output_lines:
        # change batch_output_lengths formatting
        batch_output_lengths_match = re.search(batch_output_lengths_pattern_old, line)
        batch_output_lengths_index = int(batch_output_lengths_match.group(1))
        batch_output_lengths = int(line.strip().split()[-1])
        #bmark_info[curr_iteration]['batch_output_lengths'][batch_output_lengths_index] = batch_output_lengths
        print(f'line: {line.strip()}, batch_output_lengths: {batch_output_lengths}')


def main(args):
    bmark_output_paths = args.bmark_output_paths
    nvsmi_output_paths = args.nvsmi_output_paths

    # save bmark_output_paths in *_old.out files so mistakes don't happen
    for bmark_output_path in bmark_output_paths:
        save_output_path = bmark_output_path.replace('.out', '_old.out')
        shutil.copy(bmark_output_path, save_output_path)
        format_bmark_output(bmark_output_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--bmark_output_paths',
        type=str,
        nargs='+',
        help='paths to bmark output files'
    )
    parser.add_argument(
        '--nvsmi_output_paths',
        type=str,
        nargs='+',
        help='paths to nvsmi output files'
    )
    args = parser.parse_args()
    main(args)

