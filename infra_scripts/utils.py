# Random utility functions
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
        #raise ValueError('! uh oh !')

    return result_stdout


# because I can't figure out ssh forwarding...
def ssh_and_run_command(host, command):
    ssh_command = f"ssh -A {host} {command}"
    print(f'ssh_command: {ssh_command}')
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
