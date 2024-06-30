import subprocess
import argparse
import re
import aiofiles
import asyncio
import time
import paramiko

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


def get_nvsmi_output_local():
    nvsmi_output = subprocess.check_output(['nvidia-smi'])
    decoded_output = nvsmi_output.decode('utf-8')
    return decoded_output


def get_nvsmi_output_remote(client: paramiko.SSHClient):
    stdin, nvsmi_output, stderr = client.exec_command('nvidia-smi')
    decoded_output = nvsmi_output.read().decode('utf-8')
    return decoded_output


# TODO: Need to take a look at nvsmi output parsing again
#       - more explicit timestamp matching
#       - make sure some of the matches have surrounding spaces
temp_celsius_pattern = r"\s\d+C\s"
power_usage_pattern = r"\s\d+W / \d+W\s"
memory_usage_pattern = r"\s\d+MiB / \d+MiB\s"
gpu_utilization_pattern = r"\s\d+%\s"
timestamp_pattern = r"\s([01]?[0-9]|2[0-3]):([0-5]?[0-9]):([0-5]?[0-9])\s"

def parse_nvsmi_info():

async def get_nvsmi_info(
    gpu_type: str,
    multi_node: bool,
    clients: list[paramiko.SSHClient]
):
    output = subprocess.check_output(['nvidia-smi'])
    decoded_output = output.decode('utf-8')
    nvsmi_dict = {}
    curr_GPU = -1 # TODO: this is janky
    timestamp_found = False # TODO: this is also kinda janky

    for line in decoded_output.split('\n'):
        temp_celsius_match = re.search(temp_celsius_pattern, line)
        power_usage_match = re.search(power_usage_pattern, line)
        memory_usage_match = re.search(memory_usage_pattern, line)
        gpu_utilization_match = re.search(gpu_utilization_pattern, line)
        timestamp_match = re.search(timestamp_pattern, line)

        # Timestamp only happens once and is not correlated with a specific GPU
        if timestamp_match and not timestamp_found:
            time_string = timestamp_match.group().strip()
            nvsmi_dict['timestamp_readable'] = time_string
            current_date = datetime.now().strftime("%Y-%m-%d")
            datetime_string = f"{current_date} {time_string}"
            datetime_object = datetime.strptime(datetime_string, "%Y-%m-%d %H:%M:%S")
            timestamp = datetime_object.timestamp()
            nvsmi_dict['timestamp_raw'] = timestamp
            timestamp_found = True

        # Correlate readings with a specific GPU
        if ((gpu_type == 'v10032gb' and 'V100S-PCIE-32GB' in line) or
            (gpu_type == 'a10040gb' and 'A100-SXM4-40GB' in line)):
            curr_GPU += 1
            nvsmi_dict[curr_GPU] = {}

        if temp_celsius_match:
            nvsmi_dict[curr_GPU]['temp_celsius'] = temp_celsius_match.group().strip()
        if power_usage_match:
            nvsmi_dict[curr_GPU]['power_usage'] = power_usage_match.group().strip()
        if memory_usage_match:
            nvsmi_dict[curr_GPU]['memory_usage'] = memory_usage_match.group().strip()
        if gpu_utilization_match:
            nvsmi_dict[curr_GPU]['gpu_utilization'] = gpu_utilization_match.group().strip()

    # make sure a GPU match was found in the nvidia-smi output
    assert(curr_GPU > -1)
    nvsmi_dict['num_gpus'] = curr_GPU + 1
    return nvsmi_dict


async def nvsmi_loop(
    nvsmi_filepath: str,
    host_filepath: str,
    container_filepath: str,
    container_id: str,
    gpu_type: str,
    multi_node: bool,
    clients: list[paramiko.SSHClient]
):
    while get_nvsmi_loop_running(
        host_filepath,
        container_filepath,
        container_id
    ):
        start_time = time.time()
        nvsmi_dict = await get_nvsmi_info(
            gpu_type,
            multi_node,
            clients
        )
        async with aiofiles.open(nvsmi_filepath, 'a') as f:
            await f.write(str(nvsmi_dict) + '\n')

        elapsed_time = time.time() - start_time
        sleep_duration = max(0, 1 - elapsed_time)
        await asyncio.sleep(sleep_duration)


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


def open_ssh_connections(
    remote_nodes: list[str],
    username: str
):
    clients = []
    for node in remote_nodes:
        client = paramiko.SSHClient()
        client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

        agent = paramiko.Agent()
        agent_keys = agent.get_keys()
        if len(agent_keys) == 0:
            raise Exception('No SSH keys available in the SSH agent')

        client.connect(
            node,
            username=username,
            pkey=agent_keys[0]
        )
        clients.append(client)
    return clients


def close_ssh_connections(clients: list[paramiko.SSHClient]):
    for client in clients:
        client.close()


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

    # if multi-node experiment, open ssh connections to all worker nodes
    clients = []
    if args.multi_node:
        clients = open_ssh_connections(
            args.worker_ips,
            args.username
        )
    # begin nvsmi loop until completion file is written
    asyncio.run(nvsmi_loop(
        nvsmi_filepath,
        host_filepath,
        container_filepath,
        args.container_id,
        args.gpu_type,
        args.multi_node,
        clients
    ))
    # if multi-node experiment, close ssh connections after experiment completion
    if args.multi_node:
        close_ssh_connections(clients)


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
    parser.add_argument(
        '--multi_node',
        default=False,
        action='store_true',
        help='specify whether to collect multi-node statistics for a multi-node experiment'
    )
    parser.add_argument(
        '--worker_ips',
        nargs='+',
        type=str,
        help='provide ip addresses of worker nodes for a multi-node experiment'
    )
    parser.add_argument(
        '--ssh_username',
        type=str,
        help='provide ssh username for a multi-node experiment'
    )
    args = parser.parse_args()

    if args.multi_node and (not args.worker_ips or not args.ssh_username):
        raise ValueError('multi-node experiment specified without worker ip addresses or ssh username')
    main(args)
