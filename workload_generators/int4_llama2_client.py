# Example request to Llama2 endpoint for testing and development purposes
import time
import datetime
import requests


client_side_start_time = time.time()

llama2_base_response = requests.post(
    #'http://127.0.0.1:8000/',
    'http://130.127.133.221:8000/',
    json={
        'prompt': 'What are the ingredients of olio de aglio? I do not want the entire recipe, only a list of ingredients.'
    }
)
llama2_base_response_split = llama2_base_response.text.split()

client_side_end_time = time.time()
client_side_latency = client_side_end_time - client_side_start_time
server_side_latency = float(llama2_base_response_split[-1])
num_output_tokens = int(llama2_base_response_split[-2])
#num_output_tokens = len(llama2_base_response_split) - 1

client_side_tokens_per_sec = num_output_tokens / client_side_latency
server_side_tokens_per_sec = num_output_tokens / server_side_latency

client_side_latency_readable = str(datetime.timedelta(seconds=client_side_latency))
server_side_latency_readable = str(datetime.timedelta(seconds=server_side_latency))


print(f'llama2_base_response.text: {llama2_base_response.text}')
print(f'CLIENT SIDE LATENCY: {client_side_latency_readable}')
print(f'CLIENT SIDE TOKEN/SEC: {client_side_tokens_per_sec}')
print(f'SERVER SIDE LATENCY: {server_side_latency_readable}')
print(f'SERVER SIDE TOKEN/SEC: {server_side_tokens_per_sec}')
