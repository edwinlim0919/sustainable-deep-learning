import paramiko


def get_hostnames_from_file(file_name):
    with open(file_name, 'r') as file:
        ssh_commands = file.readlines()
    
    hostnames = []
    for command in ssh_commands:
        parts = command.strip().split('@')
        if len(parts) == 2:
            username, host = parts[1].split('@')
            hostname = get_hostname(username, host)
            if hostname:
                hostnames.append(hostname)
    
    return hostnames


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
        print(f"Failed to get hostname for {host}: {e}")
        return None


if __name__ == '__main__':
    file_name = input('Enter the name of the file containing SSH commands: ')
    hostnames = get_hostnames_from_file(file_name)
    
    print('Hostnames:')
    for hostname in hostnames:
        print(hostname)
