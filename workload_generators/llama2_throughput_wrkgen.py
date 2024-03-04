import json
import random
import argparse
import requests
import numpy as np
import time
import asyncio
#import aiohttp
import aiofile
import os
import sys

from concurrent.futures import ThreadPoolExecutor


result_file_lock = asyncio.Lock()
#request_file_lock = asyncio.Lock()


# Sampling dataset prompts for throughput experiments
def sample_dataset_prompts(
    dataset_path: str,
    num_requests_sample: int
):
    with open(dataset_path, 'r', encoding='utf-8') as f:
        dataset = json.load(f)

    # Filter out the conversations with less than 2 turns.
    dataset = [data for data in dataset if len(data["conversations"]) >= 2]

    # Only keep human prompts from each conversation.
    dataset_human = []
    for data in dataset:
        for conv in data['conversations']:
            if conv['from'] == 'human':
                dataset_human.append(conv['value'])

    if num_requests_sample < 1:
        num_requests_sample = len(dataset_human)

    sampled_dataset = random.sample(dataset_human, num_requests_sample)
    return sampled_dataset


def send_request(
    prompt: str,
    head_node_ip: str
):
    client_side_start_time = time.time()
    response = requests.post(
        head_node_ip,
        json={
            'prompt': prompt
        }
    )
    client_side_end_time = time.time()

    return response, client_side_start_time, client_side_end_time



async def send_request_and_log(
    #session,
    prompt: str,
    head_node_ip: str,
    curr_rate: float,
    requests_per_rate: int,
    output_file_path: str,
    curr_dir: str
):
    print(f'SEND_REQUEST_AND_LOG START prompt: {prompt}')
    sys.stdout.flush()

    loop = asyncio.get_running_loop()
    response, client_side_start_time, client_side_end_time = await loop.run_in_executor(None, send_request, prompt, head_node_ip)

    response_split = response.text.split()
    client_side_latency = client_side_end_time - client_side_start_time
    server_side_latency = float(response_split[-1])
    num_output_tokens = int(response_split[-2])

    response_data = {
        'curr_rate' : curr_rate,
        'requests_per_rate' : requests_per_rate,
        'client_side_start_time' : client_side_start_time,
        'client_side_latency': client_side_latency,
        'server_side_latency': server_side_latency,
        'num_output_tokens': num_output_tokens,
        'response_text' : response_text
    }

    resp_file_path = curr_dir + '/' + output_file_path
    print(f'SEND_REQUEST_AND_LOG WRITE RESP: {response_data}')
    print(f'SEND_REQUEST_AND_LOG resp_file_path: {resp_file_path}')
    sys.stdout.flush()
    async with result_file_lock:
        async with aiofiles.open(resp_file_path, 'a') as respfile:
            await respfile.write(str(response_data) + '\n')


async def send_requests_rate(
    sampled_dataset: list[str],
    head_node_ip: str,
    curr_rate: float,
    requests_per_rate: int,
    output_file_path: str,
    curr_dir: str
):
    # requests per minute converted to requests per second
    lambda_rate = curr_rate / 60

    # calculating arrival times
    inter_arrival_times = np.random.exponential(1 / lambda_rate, size=requests_per_rate)
    arrival_times = np.cumsum(inter_arrival_times)

    # eliminate unnecessary waiting time for first arrival time
    initial_arrival_time_offset = arrival_times[0] * 0.9
    arrival_times = [arrival_time - initial_arrival_time_offset for arrival_time in arrival_times]
    print(f'SEND_REQUESTS_RATE lambda_rate: {lambda_rate}')
    print(f'SEND_REQUESTS_RATE arrival_times: {arrival_times}')
    sys.stdout.flush()

    start_time = time.time()
    tasks = []
    for i in range(requests_per_rate):
        send_time = start_time + arrival_times[i]
        await asyncio.sleep(max(0, send_time - time.time()))
        task = asyncio.create_task(send_request_and_log(
            sampled_dataset[i],
            head_node_ip,
            curr_rate,
            requests_per_rate,
            output_file_path,
            curr_dir
        ))
        tasks.append(task)

    await asyncio.gather(*tasks)


# Generate a slowly increasing amount of request rates according to a Poisson distribution
async def generate_requests(
    sampled_dataset: list[str],
    head_node_ip: str,
    requests_per_rate: int,         # requests
    start_rate: float,              # requests per minute
    end_rate: float,                # requests per minute
    increase_rate: float,           # requests per minute
    output_file_path: str,
    curr_dir: str
):
    # for reproducability
    np.random.seed(42)
    curr_rate = start_rate

    while curr_rate < end_rate:
        print(f'GENERATE_REQUESTS curr_rate: {curr_rate}')
        sys.stdout.flush()
        await send_requests_rate(
            sampled_dataset,
            head_node_ip,
            curr_rate,
            requests_per_rate,
            output_file_path,
            curr_dir
        )
        curr_rate = curr_rate * increase_rate


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Throughput generation script for LLM inference experiments.')
    parser.add_argument(
        '--dataset-path',
        required=True,
        type=str,
        help='The path to the dataset file.'
    )
    parser.add_argument(
        '--output-file-path',
        required=True,
        type=str,
        help='The path to the JSON output file.'
    )
    parser.add_argument(
        '--head-node-ip',
        required=True,
        type=str,
        help='The ip address of the Ray head node.'
    )
    parser.add_argument(
        '--num-requests-sample',
        required=True,
        type=int,
        help='The number of requests to sample. Specify 0 or less to sample the entire dataset.'
    )
    parser.add_argument(
        '--requests-per-rate',
        required=True,
        type=int,
        help='The number of requests to send per request rate.'
    )
    parser.add_argument(
        '--start-rate',
        required=True,
        type=float,
        help='The starting request rate in requests per minute.'
    )
    parser.add_argument(
        '--end-rate',
        required=True,
        type=float,
        help='The ending request rate in requests per minute.'
    )
    parser.add_argument(
        '--increase-rate',
        required=True,
        type=float,
        help='Request rate multiplicative increase per iteration.'
    )
    args = parser.parse_args()

    print(f'Sampling dataset {args.dataset_path}...')
    sampled_dataset = sample_dataset_prompts(
        args.dataset_path,
        args.num_requests_sample
    )
    sampled_dataset_len = len(sampled_dataset)

    curr_dir = os.getcwd()
    print('Generating requests...')
    print(f'sampled_dataset_len: {sampled_dataset_len}')
    print(f'head_node_ip: {args.head_node_ip}')
    print(f'requests_per_rate: {args.requests_per_rate}')
    print(f'start_rate: {args.start_rate}')
    print(f'end_rate: {args.end_rate}')
    print(f'increase_rate: {args.increase_rate}')
    print(f'output_file_path: {args.output_file_path}')
    print(f'curr_dir: {curr_dir}')
    #loop = asyncio.get_event_loop()
    #try:
    #    loop.run_until_complete(generate_requests(
    #        sampled_dataset,
    #        args.head_node_ip,
    #        args.requests_per_rate,
    #        args.start_rate,
    #        args.end_rate,
    #        args.increase_rate,
    #        args.output_file_path,
    #        curr_dir
    #    ))
    #finally:
    #    loop.close()
    asyncio.run(generate_requests(
        sampled_dataset,
        args.head_node_ip,
        args.requests_per_rate,
        args.start_rate,
        args.end_rate,
        args.increase_rate,
        args.output_file_path,
        curr_dir
    ))

    print('Done.')
