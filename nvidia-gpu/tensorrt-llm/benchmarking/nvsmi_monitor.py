import subprocess
import argparse
import re
import aiofiles
import asyncio

from pathlib import Path
from datetime import datetime


def get_nvsmi_loop_running(
    host_filepath: str,
    container_filepath: str,
    container_id: str
) -> bool:
    # Copy the container stop file from container to host
    container_copy(
        host_filepath,
        container_filepath,
        container_id,
        False
    )

    # check if the script in the container has signaled completion
    with open(host_filepath, 'r') as f:
        container_stop_lines = f.readlines()
    return not 'COMPLETED' in container_stop_lines[0]


temp_celsius_pattern = r"\d+C"
power_usage_pattern = r"\d+W / \d+W"
memory_usage_pattern = r"\d+MiB / \d+MiB"
gpu_utilization_pattern = r"\d+%"
timestamp_pattern = r"(?:[01]\d|2[0-3]):[0-5]\d:[0-5]\d"

#async def get_nvsmi_info_V100S_PCIE_32GB():
async def get_nvsmi_info(gpu_type: str):
    output = subprocess.check_output(['nvidia-smi'])
    decoded_output = output.decode('utf-8')
    nvsmi_dict = {}
    curr_GPU = -1 # TODO: this is janky

    for line in decoded_output.split('\n'):
        temp_celsius_match = re.search(temp_celsius_pattern, line)
        power_usage_match = re.search(power_usage_pattern, line)
        memory_usage_match = re.search(memory_usage_pattern, line)
        gpu_utilization_match = re.search(gpu_utilization_pattern, line)
        timestamp_match = re.search(timestamp_pattern, line)

        # Timestamp only happens once and is not correlated with a specific GPU
        if timestamp_match:
            nvsmi_dict['timestamp_readable'] = timestamp_match.group()
            time_string = timestamp_match.group()
            current_date = datetime.now().strftime("%Y-%m-%d")
            datetime_string = f"{current_date} {time_string}"
            datetime_object = datetime.strptime(datetime_string, "%Y-%m-%d %H:%M:%S")
            timestamp = datetime_object.timestamp()
            nvsmi_dict['timestamp_raw'] = timestamp

        # Correlate readings with a specific GPU
        if ((gpu_type == 'v10032gb' and 'V100S-PCIE-32GB' in line) or
            (gpu_type == 'a10040gb' and 'A100-SXM4-40GB' in line)):
            curr_GPU += 1
            nvsmi_dict[curr_GPU] = {}

        if temp_celsius_match:
            nvsmi_dict[curr_GPU]['temp_celsius'] = temp_celsius_match.group()
        if power_usage_match:
            nvsmi_dict[curr_GPU]['power_usage'] = power_usage_match.group()
        if memory_usage_match:
            nvsmi_dict[curr_GPU]['memory_usage'] = memory_usage_match.group()
        if gpu_utilization_match:
            nvsmi_dict[curr_GPU]['gpu_utilization'] = gpu_utilization_match.group()

    # make sure a GPU match was found in the nvidia-smi output
    assert(curr_GPU > -1)
    nvsmi_dict['num_gpus'] = curr_GPU + 1
    return nvsmi_dict


# TODO: lots of repeat code with V100 nvsmi info parsing, but tbh don't really care lol
#       wait a minute actually I do care
#async def get_nvsmi_info_A100_SXM4_40GB():
#    output = subprocess.check_output(['nvidia-smi'])
#    decoded_output = output.decode('utf-8')
#    nvsmi_dict = {}
#    curr_GPU = -1
#
#    for line in decoded_output.split('\n'):
#        timestamp_match = re.search(timestamp_pattern, line)
#
#        if timestamp_match:
#            nvsmi_dict['timestamp_readable'] = timestamp_match.group()
#            current_date = datetime.now().strftime("%Y-%m-%d")
#            datetime_string = f"{current_date} {time_string}"
#            datetime_object = datetime.strptime(datetime_string, "%Y-%m-%d %H:%M:%S")
#            timestamp = datetime_object.timestamp()
#            nvsmi_dict['timestamp_raw'] = timestamp
#
#        # correlate readings with GPU index
#        if 'A100-SXM4-40GB' in line:
#            curr_GPU += 1
#            nvsmi_dict[curr_GPU] = {}
#
#        print(line)
#
#    print()
#    for key, value in nvsmi_dict.items():
#        print('{key}: {value}')


#async def nvsmi_loop_V100S_PCIE_32GB(
async def nvsmi_loop(
    nvsmi_filepath: str,
    host_filepath: str,
    container_filepath: str,
    container_id: str,
    gpu_type: str
):
    while get_nvsmi_loop_running(
        host_filepath,
        container_filepath,
        container_id
    ):
        #nvsmi_dict = await get_nvsmi_info_V100S_PCIE_32GB()
        nvsmi_dict = await get_nvsmi_info(gpu_type)
        async with aiofiles.open(nvsmi_filepath, 'a') as f:
            await f.write(str(nvsmi_dict) + '\n')
        await asyncio.sleep(1)


def container_copy(
    host_filepath: str,
    container_filepath: str,
    container_id: str,
    to_container: bool
):
    try:
        if to_container:
            command = [
                'sudo',
                'docker',
                'cp',
                host_filepath,
                f'{container_id}:{container_filepath}'
            ]
            result = subprocess.run(command, check=True)
        else:
            command = [
                'sudo',
                'docker',
                'cp',
                f'{container_id}:{container_filepath}',
                host_filepath
            ]
            result = subprocess.run(command, check=True)
    except subprocess.CalledProcessError as e:
        print(f'container_copy error: {e}')


def container_delete(
    container_filepath: str,
    container_id: str
):
    try:
        command = [
            'sudo',
            'docker',
            'exec',
            container_id,
            'rm',
            '-f',
            container_filepath
        ]
        result = subprocess.run(command, check=True)
    except subprocess.CalledProcessError as e:
        print(f'container_delete error: {e}')


def host_delete(host_filepath: str):
    try:
        command = [
            'sudo',
            'rm',
            '-f',
            host_filepath
        ]
        result = subprocess.run(command, check=True)
    except subprocess.CalledProcessError as e:
        print(f'host_delete error: {e}')


def main(args):
    nvsmi_filepath = f'{args.output_dir}/{args.output_file}'
    host_filepath = f'{args.output_dir}/{args.container_stop_file}'
    container_filepath = f'{args.container_output_dir}/{args.container_stop_file}'

    # delete stop file if it already exists in host
    host_delete(host_filepath)
    # delete stop file if it already exists in container
    container_delete(
        container_filepath,
        args.container_id
    )

    # create output dir, output file, and container stop file
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    with (output_dir / args.output_file).open('w') as f:
        if args.gpu_type == 'v10032gb':
            f.write(f'nvsmi profiling: V100S-PCIE-32GB\n')
        elif args.gpu_type == 'a10040gb':
            f.write(f'nvsmi profiling: A100-SXM4-40GB\n')
    with (output_dir / args.container_stop_file).open('w') as f:
        f.write('RUNNING')

    # copy stop file to docker container
    container_copy(
        host_filepath,
        container_filepath,
        args.container_id,
        True
    )

    # make sure gpu_type is supported
    assert((args.gpu_type == 'v10032gb') or
           (args.gpu_type == 'a10040gb'))
    #if args.gpu_type == 'v10032gb':
    #asyncio.run(nvsmi_loop_V100S_PCIE_32GB(
    asyncio.run(nvsmi_loop(
        nvsmi_filepath,
        host_filepath,
        container_filepath,
        args.container_id,
        args.gpu_type
    ))
    #elif args.gpu_type == 'a10040gb':
    #asyncio.run(get_nvsmi_info_A100_SXM4_40GB())



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
    parser.add_argument(
        '--container_id',
        type=str,
        required=True,
        help='container id for tensorrt-llm container'
    )
    parser.add_argument(
        '--container_output_dir',
        type=str,
        required=True,
        help='directory in docker container for saving output files'
    )
    parser.add_argument(
        '--container_stop_file',
        type=str,
        required=True,
        help='filepath in docker container for coordinating nvsmi loop stop'
    )
    parser.add_argument(
        '--gpu_type',
        type=str,
        required=True,
        help='specify v10032gb or a10040gb for parsing nvsmi output'
    )
    args = parser.parse_args()
    main(args)
