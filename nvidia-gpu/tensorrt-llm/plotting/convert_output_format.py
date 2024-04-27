import argparse
import re
import ast
import os
import shutil


batch_output_lengths_pattern_old = r'batch_output_lengths\[(\d+)\]: \[(.*?)\]'

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


def format_nvsmi_output(
    nvsmi_output_path: str,
    num_gpus: int
):
    with open(nvsmi_output_path, 'r') as f:
        nvsmi_output_lines = f.readlines()

    modified_lines = []
    # random nitpick stuff
    if 'V100S_PCIE_32GB' in nvsmi_output_lines[0]:
        modified_lines.append('nvsmi profiling: V100S-PCIE-32GB\n')
    else:
        modified_lines.append(nvsmi_output_lines[0])

    for line in nvsmi_output_lines[1:]:
        nvsmi_dict = ast.literal_eval(line)

        # if 'power_usage' is a key in the dictionary (instead of multiple entries for multiple GPUs), then it is the old format
        if 'power_usage' in nvsmi_dict:
            new_nvsmi_dict = {
                'timestamp_readable': nvsmi_dict['timestamp_readable'],
                'timestamp_raw': nvsmi_dict['timestamp_raw']
            }
            for i in range(num_gpus):
                if i == 0:
                    new_nvsmi_dict[i] = {
                        'temp_celsius': 'N/A',
                        'power_usage': nvsmi_dict['power_usage'],
                        'memory_usage': nvsmi_dict['memory_usage'],
                        'gpu_utilization': nvsmi_dict['gpu_utilization']
                    }
                else:
                    new_nvsmi_dict[i] = {
                        'temp_celsius': 'N/A',
                        'power_usage': 'N/A',
                        'memory_usage': 'N/A',
                        'gpu_utilization': 'N/A'
                    }

            # formatting nitpick, adding last entry here
            new_nvsmi_dict['num_gpus'] = num_gpus 
            modified_lines.append(str(new_nvsmi_dict) + '\n')
        else:
            # original line is fine
            modified_lines.append(line)

    with open(nvsmi_output_path, 'w') as f:
        f.writelines(modified_lines)



def main(args):
    bmark_output_paths = args.bmark_output_paths
    nvsmi_output_paths = args.nvsmi_output_paths
    num_gpus = args.num_gpus

    if nvsmi_output_paths and not num_gpus:
        raise ValueError('need to supply --num_gpus when reformatting --nvsmi_output_paths')

    # save bmark_output_paths in *_old.out files before converting original file to new format
    if bmark_output_paths:
        for bmark_output_path in bmark_output_paths:
            save_output_path = bmark_output_path.replace('.out', '_old.out')
            shutil.copy(bmark_output_path, save_output_path)
            format_bmark_output(bmark_output_path)

    # save nvsmi_output_paths in *._old.out files before converting original file ot new format
    if nvsmi_output_paths:
        for nvsmi_output_path in nvsmi_output_paths:
            save_output_path = nvsmi_output_path.replace('.out', '_old.out')
            shutil.copy(nvsmi_output_path, save_output_path)
            format_nvsmi_output(nvsmi_output_path, num_gpus)


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
    parser.add_argument(
        '--num_gpus',
        type=int,
        help='number of gpus for reformatting old nvsmi data'
    )
    args = parser.parse_args()
    main(args)
