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

        # Some files might already be in the correct format
        if batch_output_lengths_match:
            batch_output_lengths_index = int(batch_output_lengths_match.group(1))
            length_list_str = batch_output_lengths_match.group(2)
            batch_output_lengths = int(ast.literal_eval(f'[{length_list_str}]')[0])
            new_line = f'batch_output_lengths[{batch_output_lengths_index}]: {batch_output_lengths}\n'
            modified_lines.append(new_line)
        else:
            # no modification needed
            modified_lines.append(line)

    with open(bmark_output_path, 'w') as f:
        f.writelines(modified_lines)


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

