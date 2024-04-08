import subprocess
import argparse
import re
import time


power_usage_pattern = r"\d+W / \d+W"
memory_usage_pattern = r"\d+MiB / \d+MiB"
gpu_utilization_pattern = r"\d+%"
timestamp_pattern = r"(?:[01]\d|2[0-3]):[0-5]\d:[0-5]\d"


def get_nvsmi_info_V100S_PCIE_32GB():
    output = subprocess.check_output(['nvidia-smi'])
    decoded_output = output.decode('utf-8')
    nvsmi_dict = {}

    for line in decoded_output.split('\n'):
        power_usage_match = re.search(power_usage_pattern, line)
        memory_usage_match = re.search(memory_usage_pattern, line)
        gpu_utilization_match = re.search(gpu_utilization_pattern, line)
        timestamp_match = re.search(timestamp_pattern, line)

        if timestamp_match:
            nvsmi_dict['timestamp'] = timestamp_match.group()
        if power_usage_match:
            nvsmi_dict['power_usage'] = power_usage_match.group()
        if memory_usage_match:
            nvsmi_dict['memory_usage'] = memory_usage_match.group()
        if gpu_utilization_match:
            nvsmi_dict['gpu_utilization'] = gpu_utilization_match.group()

    #print(nvsmi_dict)

    #timestamp = time.time()
    #local_time = time.localtime(timestamp)
    #formatted_time = time.strftime("%H:%M:%S", local_time)
    #print(timestamp)
    #print(local_time)
    #print(formatted_time)
    #print(output)


def main(args):
    # create output dir and file
    #with (args.output_dir / args.output_file).open('w') as f:
    #    f.write(f'engine path: {args.engine_dir}\n')
    #    f.write(f'tokenizer path: {args.tokenizer_dir}\n')
    get_nvsmi_info_V100S_PCIE_32GB()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--output_dir',
        type=str,
        required=True,
        help='directory for saving output files'
    )
    parser.add_argument(
        '--output_file',
        type=str,
        required=True,
        help='output file name'
    )
    args = parser.parse_args()
    main(args)
