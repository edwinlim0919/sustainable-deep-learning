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


batch_input_tokens_pattern = r'batch_input_tokens\[(\d+)\]: \[(.*?)\]'
batch_output_tokens_pattern = r'batch_output_tokens\[(\d+)\]: \[(.*?)\]'

batch_input_lengths_pattern = r'batch_input_lengths\[(\d+)\]'
#batch_output_lengths_pattern = r'batch_output_lengths\[(\d+)\]'
batch_output_lengths_pattern = r'batch_output_lengths\[(\d+)\]: \[(.*?)\]'


def parse_bmark_output(bmark_output_path):
    with open(bmark_output_path, 'r') as f:
        bmark_output_lines = f.readlines()

    engine_path_line = bmark_output_lines[0]
    tokenizer_path_line = bmark_output_lines[1]
    num_iterations_line = bmark_output_lines[2]
    num_iterations = int(num_iterations_line.split()[-1])
    bmark_info = {}

    for line in bmark_output_lines[3:]:
        if 'iteration' in line:
            curr_iteration = int(line.strip().split()[-1])
            bmark_info[curr_iteration] = {}
            bmark_info[curr_iteration]['batch_input_tokens'] = {}
            bmark_info[curr_iteration]['batch_input_lengths'] = {}
            bmark_info[curr_iteration]['batch_output_tokens'] = {}
            bmark_info[curr_iteration]['batch_output_lengths'] = {}
        if 'batch_size' in line:
            bmark_info[curr_iteration]['batch_size'] = int(line.strip().split()[-1])
        if 'max_input_tokens' in line:
            bmark_info[curr_iteration]['max_input_tokens'] = int(line.strip().split()[-1])
        if 'max_output_tokens' in line:
            bmark_info[curr_iteration]['max_output_tokens'] = int(line.strip().split()[-1])
        if 'batch_start_time' in line:
            bmark_info[curr_iteration]['batch_start_time'] = float(line.strip().split()[-1])
        if 'batch_end_time' in line:
            bmark_info[curr_iteration]['batch_end_time'] = float(line.strip().split()[-1])
        if 'batch_input_tokens' in line:
            batch_input_tokens_match = re.search(batch_input_tokens_pattern, line)
            batch_input_tokens_index = int(batch_input_tokens_match.group(1))
            token_list_str = batch_input_tokens_match.group(2)
            batch_input_tokens_list = ast.literal_eval(f'[{token_list_str}]')
            bmark_info[curr_iteration]['batch_input_tokens'][batch_input_tokens_index] = batch_input_tokens_list
        if 'batch_input_lengths' in line:
            batch_input_lengths_match = re.search(batch_input_lengths_pattern, line)
            batch_input_lengths_index = int(batch_input_lengths_match.group(1))
            batch_input_lengths = int(line.strip().split()[-1])
            bmark_info[curr_iteration]['batch_input_lengths'][batch_input_lengths_index] = batch_input_lengths
        if 'batch_output_tokens' in line:
            batch_output_tokens_match = re.search(batch_output_tokens_pattern, line)
            batch_output_tokens_index = int(batch_output_tokens_match.group(1))
            token_list_str = batch_output_tokens_match.group(2)
            batch_output_tokens_list = ast.literal_eval(f'[{token_list_str}]')
            bmark_info[curr_iteration]['batch_output_tokens'][batch_output_tokens_index] = batch_output_tokens_list
        if 'batch_output_lengths' in line:
            batch_output_lengths_match = re.search(batch_output_lengths_pattern, line)
            batch_output_lengths_index = int(batch_output_lengths_match.group(1))
            #batch_output_lengths = int(line.strip().split()[-1]) # TODO: Make experiment write just the int without the list wrapper
            length_list_str = batch_output_lengths_match.group(2)
            batch_output_lengths = int(ast.literal_eval(f'[{length_list_str}]')[0])
            bmark_info[curr_iteration]['batch_output_lengths'][batch_output_lengths_index] = batch_output_lengths

    return bmark_info

    #print(bmark_info[0])
    #for key, value in bmark_info[99].items():
    #    print(f'{key}: {value}')

    #for i in range(num_iterations):
        
def parse_nvsmi_output(nvsmi_output_path):
    with open(nvsmi_output_path, 'r') as f:
        nvsmi_output_lines = f.readlines()

    hardware_platform_line = nvsmi_output_lines[0] # TODO: currently unused
    print(hardware_platform_line)

    for line in nvsmi_output_lines[1:]:
        nvsmi_dict = ast.literal_eval(line)
        print(nvsmi_dict)


def main(args):
    bmark_output_paths = args.bmark_output_paths
    nvsmi_output_paths = args.nvsmi_output_paths
    bmark_info = args.bmark_info
    assert(len(bmark_output_paths) == len(nvsmi_output_paths) and 
           len(nvsmi_output_paths) == len(bmark_info))

    for i in range(len(bmark_output_paths)):
        #bmark_output_path = bmark_output_paths[i]
        #nvsmi_output_path = nvsmi_output_paths[i]
        #print(f'BMARK: {bmark_output_path}')
        #print(f'NVSMI: {nvsmi_output_path}')
        curr_bmark_info = bmark_info[i].split()
        model_size_GB = int(curr_bmark_info[0])
        batch_size = int(curr_bmark_info[1])
        max_sequence_length = int(curr_bmark_info[2])

    parse_nvsmi_output(nvsmi_output_paths[2])
    #parse_bmark_output(bmark_output_paths[2])
    #print(bmark_output_paths[2])


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
        '--bmark_info',
        type=str,
        nargs='+',
        required=True,
        help='[model size] [batch size] [max sequence length]'
    )
    args = parser.parse_args()
    main(args)
