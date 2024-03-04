import json
import random
import argparse
import requests
import numpy as np
import time
import asyncio
import aiohttp
import aiofiles


#global_request_data = []
#global_request_data_lock = asyncio.Lock()
result_file_lock = asyncio.Lock()


# Sampling dataset prompts for throughput experiments
def sample_dataset_prompts(
    dataset_path: str,
    num_requests_sample: int
):
    with open(dataset_path, 'r', encoding='utf-8') as f:
        dataset = json.load(f)

    # Filter out the conversations with less than 2 turns.
    dataset = [data for data in dataset if len(data["conversations"]) >= 2]
    # Only keep the first prompt of each conversation.
    dataset = [data["conversations"][0]["value"] for data in dataset]
    if num_requests_sample < 1:
        num_requests_sample = len(dataset)

    sampled_dataset = random.sample(dataset, num_requests_sample)
    return sampled_dataset


async def send_request(
    session,
    prompt: str,
    head_node_ip: str,
    curr_rate: float,
    requests_per_rate: int,
    output_file_path: str
):
    client_side_start_time = time.time()

    async with session.post(head_node_ip, json={'prompt': prompt}) as response:
        print(f"Request sent: Status Code: {response.status}")
        response_text = await response.text()

    client_side_end_time = time.time()
    response_split = response_text.split()
    client_side_latency = client_side_end_time - client_side_start_time
    server_side_latency = float(response_split[-1])
    num_output_tokens = int(response_split[-2])

    request_data = {
        'curr_rate' : curr_rate,
        'requests_per_rate' : requests_per_rate,
        'client_side_latency': client_side_latency,
        'server_side_latency': server_side_latency,
        'num_output_tokens': num_output_tokens
    }
    #print(f'REQUEST_DATA: {request_data}')

    #async with global_request_data_lock:
    #    global_request_data.append(request_data)
    async with result_file_lock:
        async with aiofiles.open(output_file_path, 'a') as outfile:
        #with open(output_file_path, 'a') as outfile:
            #json.dump(global_request_data, outfile, indent=4)
            await outfile.write(str(request_data) + '\n')


async def send_requests_rate(
    sampled_dataset: list[str],
    head_node_ip: str,
    curr_rate: float,
    requests_per_rate: int,
    output_file_path: str
):
    # requests per minute converted to requests per second
    lambda_rate = curr_rate / 60

    # calculating arrival times
    inter_arrival_times = np.random.exponential(1 / lambda_rate, size=requests_per_rate)
    arrival_times = np.cumsum(inter_arrival_times)
    start_time = time.time()

    async with aiohttp.ClientSession() as session:
        tasks = []
        for i in range(requests_per_rate):
            send_time = start_time + arrival_times[i]
            await asyncio.sleep(max(0, send_time - time.time()))
            task = asyncio.create_task(send_request(
                session,
                sampled_dataset[i],
                head_node_ip,
                curr_rate,
                requests_per_rate,
                output_file_path
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
    output_file_path: str
):
    # for reproducability
    np.random.seed(42)
    curr_rate = start_rate

    while curr_rate < end_rate:
        print(f'Request rate: {curr_rate}...')
        await send_requests_rate(
            sampled_dataset,
            head_node_ip,
            curr_rate,
            requests_per_rate,
            output_file_path
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

    print('Generating requests...')
    print(f'sampled_dataset_len: {sampled_dataset_len}')
    print(f'requests_per_rate: {args.requests_per_rate}')
    print(f'start_rate: {args.start_rate}')
    print(f'end_rate: {args.end_rate}')
    print(f'increase_rate: {args.increase_rate}')
    asyncio.run(generate_requests(
        sampled_dataset,
        #args.head_node_ip,
        'http://130.127.133.221:8000/',
        args.requests_per_rate,
        args.start_rate,
        args.end_rate,
        args.increase_rate,
        args.output_file_path
    ))

    #print('Writing results...')
    #with open(args.output_file_path, 'w') as outfile:
    #    json.dump(global_request_data, outfile, indent=4)
    print('Done.')
