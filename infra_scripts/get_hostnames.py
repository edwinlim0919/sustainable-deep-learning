import paramiko
import argparse


def get_hostnames_from_file(file_name):
    with open(file_name, 'r') as file:
        ssh_commands = file.readlines()
    
    hostnames = []
    for command in ssh_commands:
        parts = command.strip().split('@')
        username = parts[0].split()[1]
        host = parts[1]

        hostname = get_hostname(username, host)
        if hostname:
            hostnames.append(hostname)
    
    return hostnames, username


def get_hostname(username, host):
    try:
        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        ssh.connect(host, username=username)
        stdin, stdout, stderr = ssh.exec_command('hostname')
        hostname = stdout.read().decode().strip()
        ssh.close()
        return hostname
    except Exception as e:
        print(f'Failed to get hostname for {host}: {e}')
        return None


def write_hostnames_to_file(hostnames, outfile, username):
    with open(outfile, 'w') as file:
        file.write(f'{username}\n')

        for hostname in hostnames:
            file.write(f'{hostname}\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Get hostnames from a list of SSH commands.')
    parser.add_argument('--ssh-list', required=True, help='The filename containing SSH commands.')
    parser.add_argument('--outfile', type=str, required=True, help='File to write the hostnames to.')
    args = parser.parse_args()

    hostnames, username = get_hostnames_from_file(args.ssh_list)
    write_hostnames_to_file(hostnames, args.outfile, username)
    print(f'Username and hostnames have been written to {args.outfile}.')
