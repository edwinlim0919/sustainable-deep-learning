# Connects ray clusters for running inference experiments
import subprocess
import argparse
import paramiko
import re
import os

import utils


def connect_worker_node(username, host, worker_connect_command):
    curr_dir = os.getcwd()
    local_script_path = f'{curr_dir}/worker_connect.sh'
    remote_script_path = f'/users/{username}/worker_connect.sh'
    with open(local_script_path, 'w') as file:
        file.write("#!/bin/bash\n")
        file.write("conda activate intel-transformers\n")
        file.write(worker_connect_command)
    os.chmod(local_script_path, 0o755)

    try:
        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        ssh.connect(host, username=username)

        sftp = ssh.open_sftp()
        sftp.put(
            local_script_path,
            remote_script_path
        )
        sftp.close()

        chmod_command = f'chmod +x {remote_script_path}'
        stdin, stdout, stderr = ssh.exec_command(chmod_command)

        execute_remote_script_command = f'{remote_script_path}'
        print(f'execute_remote_script_command: {execute_remote_script_command}')
        stdin, stdout, stderr = ssh.exec_command(execute_remote_script_command)
        worker_stdout = stdout.read().decode().strip()
        print(worker_stdout)
        ssh.close()
    except Exception as e:
        print(f'Failed to connect worker for {host}: {e}')


def connect_worker_nodes(ssh_list_file, worker_connect_command):
    with open(ssh_list_file, 'r') as file:
        ssh_commands = file.readlines()

    for command in ssh_commands:
        parts = command.strip().split('@')
        username = parts[0].split()[1]
        host = parts[1]
        connect_worker_node(username, host, worker_connect_command)


def start_head_node():
    start_cmd = 'ray start --head --port=6379'
    pattern = r"ray start --address='([^']+)'"
    result_stdout = utils.execute_cmd(start_cmd)

    match = re.search(pattern, result_stdout)
    worker_connect_command = match.group(0)
    return worker_connect_command


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Connects worker nodes for running inference experiments.')
    parser.add_argument(
        '--ssh-list',
        required=True,
        help='The filename containing SSH commands.'
    )
    args = parser.parse_args()

    worker_connect_command = start_head_node()
    #full_connect_command = f'conda activate intel-transformers && {worker_connect_command}'
    print(f'worker_connect_command: {worker_connect_command}')
    #print(f'full_connect_command: {full_connect_command}')

    connect_worker_nodes(args.ssh_list, worker_connect_command)
