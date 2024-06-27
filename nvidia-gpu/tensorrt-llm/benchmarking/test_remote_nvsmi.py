import paramiko


hostname = '130.127.134.35'
username = 'edwinlim'

client = paramiko.SSHClient()
client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

agent = paramiko.Agent()
agent_keys = agent.get_keys()
if len(agent_keys) == 0:
    raise Exception('No SSH keys available in the SSH agent')

client.connect(hostname, username=username, pkey=agent_keys[0])

print('Paramiko ssh connection success!')
