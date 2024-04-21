import subprocess
import argparse
import re
import aiofiles
import asyncio

from pathlib import Path
from datetime import datetime


temp_celsius_pattern = r"\d+C"
power_usage_pattern = r"\d+W / \d+W"
memory_usage_pattern = r"\d+MiB / \d+MiB"
gpu_utilization_pattern = r"\d+%"
timestamp_pattern = r"(?:[01]\d|2[0-3]):[0-5]\d:[0-5]\d"


#nvsmi_loop_running = True

#def get_nvsmi_loop_running() -> bool:
#    return nvsmi_loop_running


#def set_nvsmi_loop_running(value: bool):
#    global nvsmi_loop_running
#    nvsmi_loop_running = value


async def get_nvsmi_info_V100S_PCIE_32GB():
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
        if 'V100S-PCIE-32GB' in line: # TODO: this is janky
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

    #for key, value in nvsmi_dict.items():
    #    print(f'{key}: {value}')

    return nvsmi_dict


#async def nvsmi_loop_V100S_PCIE_32GB(filepath: str):
async def nvsmi_loop_V100S_PCIE_32GB(
    host_filepath: str,
    container_filepath: str,
    container_id: str
):
    while get_nvsmi_loop_running():
        nvsmi_dict = await get_nvsmi_info_V100S_PCIE_32GB()
        async with aiofiles.open(filepath, 'a') as f:
            await f.write(str(nvsmi_dict) + '\n')
        await asyncio.sleep(1)2


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


def main(args):
    # create output dir, output file, and container stop file
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    with (output_dir / args.output_file).open('w') as f:
        f.write(f'nvsmi profiling: V100S_PCIE_32GB\n')
    with (output_dir / args.container_stop_file).open('w') as f:
        f.write('RUNNING')

    # copy stop file to docker container
    host_filepath = f'{args.output_dir}/{args.container_stop_file}'
    container_filepath = f'{args.container_output_dir}/{args.container_stop_file}'
    container_copy(
        host_filepath,
        container_filepath,
        args.container_id,
        True
    )
    #filepath = args.output_dir + '/' + args.output_file
    #asyncio.run(nvsmi_loop_V100S_PCIE_32GB(filepath))
    asyncio.run(nvsmi_loop_V100S_PCIE_32GB(
        host_filepath,
        container_filepath,
        args.container_id
    ))


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
    args = parser.parse_args()
    main(args)
