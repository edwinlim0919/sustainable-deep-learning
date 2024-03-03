# Sets up ray clusters for running inference experiments
import subprocess
import argparse
import paramiko
import re
import os
import concurrent.futures

import utils


def setup_worker_node(username, host):
    try:
        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        ssh.connect(host, username=username)

        # copy setup files
        sftp = ssh.open_sftp()
        curr_dir = os.getcwd()
        local_script_path = f'{curr_dir}/worker_setup.sh'
        remote_script_path = f'/users/{username}/worker_setup.sh'
        sftp.put(
            local_script_path,
            remote_script_path
        )
        local_reqs_path = f'{curr_dir}/../intel-transformers-cpu/requirements.txt'
        remote_reqs_path = f'/users/{username}/requirements.txt'
        sftp.put(
            local_reqs_path,
            remote_reqs_path
        )
        ssh_config_path = f'/users/{username}/.ssh/config'
        sftp.put(
            ssh_config_path,
            ssh_config_path
        )
        sftp.close()
        ssh.close()

        # ugh
        remote_host = f'{username}@{host}'
        rm_command = f'rm -rf /users/{username}/neural-speed'
        clone_command = 'git clone git@github.com:edwinlim0919/neural-speed.git'
        utils.ssh_and_run_command(remote_host, rm_command)
        utils.ssh_and_run_command(remote_host, clone_command)

        # setup base dependencies
        ssh.connect(host, username=username)
        command = f'chmod +x {remote_script_path} && yes | {remote_script_path}'
        stdin, stdout, stderr = ssh.exec_command(command)

        exit_status = stdout.channel.recv_exit_status()
        if exit_status == 0:
            print("Script executed successfully")
            print(stdout.read().decode())
        else:
            print("Error executing script")
            print(stderr.read().decode())

        ssh.close()
    except Exception as e:
        print(f'Failed to setup worker for {host}: {e}')


def setup_worker_nodes(ssh_list_file):
    with open(ssh_list_file, 'r') as file:
        ssh_commands = file.readlines()

    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = []
        for command in ssh_commands:
            parts = command.strip().split('@')
            username = parts[0].split()[1]
            host = parts[1]
            future = executor.submit(setup_worker_node, username, host)
            futures.append(future)

        for future in concurrent.futures.as_completed(futures):
            try:
                future.result()
            except Exception as e:
                print(f"Error during setup: {e}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Installs dependencies for running inference experiments.')
    parser.add_argument(
        '--ssh-list',
        required=True,
        help='The filename containing SSH commands.'
    )
    args = parser.parse_args()
    setup_worker_nodes(args.ssh_list)
