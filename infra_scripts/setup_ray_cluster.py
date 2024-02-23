# Sets up ray clusters for running inference experiments
import subprocess
import argparse
import paramiko
import re
import os


def execute_cmd(cmd):
    result = subprocess.run(
        cmd,
        shell=True,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )

    if result.returncode == 0:
        result_stdout = result.stdout.decode()
        print(f'Command executed successfully: {cmd}')
        print(result_stdout)
    else:
        print('Error executing command: {cmd}')
        print(result.stderr.decode())

    pattern = r"ray start --address='([^']+)'"
    match = re.search(pattern, result_stdout)
    worker_connect_command = match.group(0)
    return worker_connect_command


def connnect_worker_node(username, host, worker_connect_command):
    try:
        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        ssh.connect(host, username=username)
        stdin, stdout, stderr = ssh.exec_command(worker_connect_command)
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
        connect_worker_node(username, host)


def start_head_node():
    start_cmd = 'ray start --head --port=6379'
    return execute_cmd(start_cmd)


# because I can't figure out ssh forwarding...
def ssh_and_run_command(host, command):
    ssh_command = f"ssh -A {host} {command}"
    try:
        result = subprocess.run(
            ssh_command,
            shell=True,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
    except subprocess.CalledProcessError as e:
        print(f"Error executing command: {command}")


def setup_worker_node(username, host):
    try:
        # ugh
        remote_host = f'{username}@{host}'
        rm_command = 'rm -rf neural-speed'
        clone_command = 'git clone git@github.com:edwinlim0919/neural-speed.git'
        ssh_and_run_command(remote_host, rm_command)
        ssh_and_run_command(remote_host, clone_command)

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
        key_path = f'/users/{username}/.ssh/id_rsa'
        sftp.put(
            key_path,
            key_path
        )
        config_path = f'/users/{username}/.ssh/config'
        sftp.put(
            config_path,
            config_path
        )
        local_reqs_path = f'{curr_dir}/../intel-transformers-cpu/requirements.txt'
        remote_reqs_path = f'/users/{username}/requirements.txt'
        sftp.put(
            local_reqs_path,
            remote_reqs_path
        )
        sftp.close()
        ssh.close()

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
        print(f'Failed to connect worker for {host}: {e}')


def setup_worker_nodes(ssh_list_file):
    with open(ssh_list_file, 'r') as file:
        ssh_commands = file.readlines()

    for command in ssh_commands:
        parts = command.strip().split('@')
        username = parts[0].split()[1]
        host = parts[1]
        setup_worker_node(username, host)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Sets up ray clusters for running inference experiments.')
    parser.add_argument(
        '--ssh-list',
        required=True,
        help='The filename containing SSH commands.'
    )
    args = parser.parse_args()
    setup_worker_nodes(args.ssh_list)

    #worker_connect_command = start_head_node()
    #print(f'worker_connect_command: {worker_connect_command}')

    #connect_worker_nodes(args.ssh_list, worker_connect_command)
