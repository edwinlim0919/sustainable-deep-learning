import argparse
import re
import ast

from pathlib import Path


max_batch_size_pattern = r"'max_batch_size': (\d+)"
iteration_pattern = r"'iteration': (\d+)"
max_input_tokens_pattern = r"'max_input_tokens': (\d+)"
max_output_tokens_pattern = r"'max_output_tokens': (\d+)"
batch_input_lengths_pattern = r"'batch_input_lengths': \[([\d, ]+)\]"

def parse_bmark_output(bmark_output_path):
    with open(bmark_output_path, 'r') as f:
        bmark_output_lines = f.readlines()

    eng_path_line = bmark_output_lines[0]
    tok_path_line = bmark_output_lines[1]
    num_iter_line = bmark_output_lines[2]

    current_dict_line = ''
    for line in bmark_output_lines[3:]:
        stripped = line.strip()
        if line[0] == '{'


        print(f'line: {line}')
        max_batch_size = re.search(max_batch_size_pattern, line).group()
        iteration = re.search(iteration_pattern, line).group()
        max_input_tokens = re.search(max_input_tokens_pattern, line).group()
        max_output_tokens = re.search(max_output_tokens_pattern, line).group()
        batch_input_lengths = re.search(batch_input_lengths_pattern, line).group()

        print(f'max_batch_size: {max_batch_size}')
        print(f'iteration: {iteration}')
        print(f'max_input_tokens: {max_input_tokens}')
        print(f'max_output_tokens: {max_output_tokens}')
        print(f'batch_input_lengths: {batch_input_lengths}')
        print()
        #dict_line = line.strip()
        #print(dict_line)

        #dict_obj = ast.literal_eval(dict_line)
        #print(f'dict_obj: {dict_obj}')

    print(bmark_output_lines[-1])


def main(args):
    bmark_output_paths = args.bmark_output_paths.split()
    nvsmi_output_paths = args.nvsmi_output_paths.split()
    assert(len(bmark_output_paths) == len(nvsmi_output_paths))

    for i in range(len(bmark_output_paths)):
        bmark_output_path = bmark_output_paths[i]
        nvsmi_output_path = nvsmi_output_paths[i]
        print(f'PATHS: {bmark_output_path} {nvsmi_output_path}')

    print(bmark_output_paths[1])
    parse_bmark_output(bmark_output_paths[1])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--bmark_output_paths',
        type=str,
        required=True,
        help='paths to bmark output files'
    )
    parser.add_argument(
        '--nvsmi_output_paths',
        type=str,
        required=True,
        help='paths to nvsmi output files'
    )
    args = parser.parse_args()
    main(args)
