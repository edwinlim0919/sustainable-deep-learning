import paramiko
import argparse
import re


def get_local_ips_from_file(file_name):
    with open(file_name, 'r') as file:
        ssh_commands = file.readlines()
    
    local_ips = []
    hosts = []
    for command in ssh_commands:
        parts = command.strip().split('@')
        username = parts[0].split()[1]
        host = parts[1]
        hosts.append(host)

        local_ip = get_local_ip(username, host)
        if local_ip:
            local_ips.append(local_ip)
    
    if len(local_ips) != len(hosts):
        raise ValueError('! uh oh !')

    return local_ips, hosts, username


def get_local_ip(username, host):
    try:
        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        ssh.connect(host, username=username)
        stdin, stdout, stderr = ssh.exec_command('hostname -I')
        ip_addresses = stdout.read().decode().strip()
        local_ip = pick_local_regex(ip_addresses)
        ssh.close()
        return local_ip
    except Exception as e:
        print(f'Failed to get local_ip for {host}: {e}')
        return None


def pick_local_regex(ip_addresses):
    pattern = r'\b10\.\d{1,3}\.\d{1,3}\.\d{1,3}\b'
    matches = re.findall(pattern, ip_addresses)
    if matches:
        return matches[0]
    else:
        return None


def write_local_ips_to_file(local_ips, hosts, outfile, username):
    with open(outfile, 'w') as file:
        file.write(f'{username}\n')

        for i in range(len(local_ips)):
            local_ip = local_ips[i]
            host = hosts[i]
            file.write(f'{local_ip} {host}\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Get local_ips from a list of SSH commands.')
    parser.add_argument('--ssh-list', required=True, help='The filename containing SSH commands.')
    parser.add_argument('--outfile', type=str, required=True, help='File to write the local_ips to.')
    args = parser.parse_args()

    local_ips, hosts, username = get_local_ips_from_file(args.ssh_list)
    write_local_ips_to_file(local_ips, hosts, args.outfile, username)
    print(f'Username and local_ips have been written to {args.outfile}.')
