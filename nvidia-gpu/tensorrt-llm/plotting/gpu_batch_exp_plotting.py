import argparse
import re
import ast

from pathlib import Path


#max_batch_size_pattern = r"'max_batch_size': (\d+)"
#iteration_pattern = r"'iteration': (\d+)"
#max_input_tokens_pattern = r"'max_input_tokens': (\d+)"
#max_output_tokens_pattern = r"'max_output_tokens': (\d+)"
#batch_input_lengths_pattern = r"'batch_input_lengths': \[([\d, ]+)\]"
#
#def parse_bmark_output(bmark_output_path):
#    with open(bmark_output_path, 'r') as f:
#        bmark_output_lines = f.readlines()
#
#    eng_path_line = bmark_output_lines[0]
#    tok_path_line = bmark_output_lines[1]
#    num_iter_line = bmark_output_lines[2]
#
#    current_dict_line = ''
#    for line in bmark_output_lines[3:]:
#        stripped = line.strip()
#        #print(stripped)
#        if stripped[0] == '{':
#            current_dict_line = stripped
#
#        if stripped[-1] != '}':
#            current_dict_line += stripped
#            continue
#        else:
#            current_dict_line += stripped
#            # this is the full result dict
#
#            max_batch_size = re.search(max_batch_size_pattern, current_dict_line).group()
#            iteration = re.search(iteration_pattern, current_dict_line).group()
#            max_input_tokens = re.search(max_input_tokens_pattern, current_dict_line).group()
#            max_output_tokens = re.search(max_output_tokens_pattern, current_dict_line).group()
#            batch_input_lengths = re.search(batch_input_lengths_pattern, current_dict_line).group()
#
#            print(f'max_batch_size: {max_batch_size}')
#            print(f'iteration: {iteration}')
#            print(f'max_input_tokens: {max_input_tokens}')
#            print(f'max_output_tokens: {max_output_tokens}')
#            print(f'batch_input_lengths: {batch_input_lengths}')
#            print()
#
#    print(bmark_output_lines[-1])


batch_input_tokens_pattern = 'batch_input_tokens\[(\d+)\]'
batch_output_tokens_pattern = 'batch_output_tokens\[(\d+)\]'


def parse_bmark_output(bmark_output_path):
    with open(bmark_output_path, 'r') as f:
        bmark_output_lines = f.readlines()

    #print(bmark_output_lines)

    engine_path_line = bmark_output_lines[0]
    tokenizer_path_line = bmark_output_lines[1]
    num_iterations_line = bmark_output_lines[2]

    num_iterations = int(num_iterations_line.split()[-1])
    print(f'num_iterations: {num_iterations}')
    bmark_info = {}

    for line in bmark_output_lines[3:]:
        if 'iteration' in line:
            curr_iteration = int(line.strip().split()[-1])
            #print(line.strip())
            #print(curr_iteration)
            bmark_info[curr_iteration] = {}
            bmark_info[curr_iteration]['batch_input_tokens'] = {}           
        if 'batch_size' in line:
            bmark_info[curr_iteration]['batch_size'] = int(line.strip().split()[-1])
        if 'max_input_tokens' in line:
            bmark_info[curr_iteration]['max_input_tokens'] = int(line.strip().split()[-1])
        if 'max_output_tokens' in line:
            bmark_info[curr_iteration]['max_output_tokens'] = int(line.strip().split()[-1])
        if 'batch_start_time' in line:
            bmark_info[curr_iteration]['batch_start_time'] = float(line.strip().split()[-1])
        if 'batch_end_time' in line:
            bmark_info[curr_tieration]['batch_end_time'] = float(line.strip().split()[-1])
        if 'batch_input_tokens' in line:
            batch_input_tokens_match = re.search(batch_input_tokens_pattern, line)
            index_number = match.group(1)



    #for i in range(num_iterations):
        


def main(args):
    bmark_output_paths = args.bmark_output_paths#.split()
    nvsmi_output_paths = args.nvsmi_output_paths#.split()
    assert(len(bmark_output_paths) == len(nvsmi_output_paths))

    for i in range(len(bmark_output_paths)):
        bmark_output_path = bmark_output_paths[i]
        nvsmi_output_path = nvsmi_output_paths[i]
        print(f'BMARK: {bmark_output_path}')
        print(f'NVSMI: {nvsmi_output_path}')

    parse_bmark_output(bmark_output_paths[2])
    print(bmark_output_paths[2])


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
    args = parser.parse_args()
    main(args)
