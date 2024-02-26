import json
import random
import argparse
import requests
import numpy as np
import time
import asyncio
import aiohttp


# Sampling dataset prompts for throughput experiments
def sample_dataset_prompts(
    dataset_path: str,
    num_requests: int
):
    with open(dataset_path, 'r', encoding='utf-8') as f:
        dataset = json.load(f)

    # Filter out the conversations with less than 2 turns.
    dataset = [data for data in dataset if len(data["conversations"]) >= 2]
    # Only keep the first prompt of each conversation.
    dataset = [data["conversations"][0]["value"] for data in dataset]
    if num_requests < 1:
        num_requests = len(dataset)

    print('SAMPLED DATASET')
    sampled_dataset = random.sample(dataset, num_requests)
    print(sampled_dataset)

    return sampled_dataset


async def send_request(session, prompt):
    async with session.post('http://127.0.0.1:8000/', json={'prompt': prompt}) as response:
        print(f"Request sent: Status Code: {response.status}")
        # TODO: Accounting stuff


async def send_requests_rate(
    sampled_dataset: List[str],
    curr_rate: int,
    requests_per_rate: int
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
            task = asyncio.create_task(send_request(session, sampled_dataset[i]))
            tasks.append(task)

        await asyncio.gather(*tasks)


# Generate a slowly increasing amount of request rates according to a Poisson distribution
def generate_requests(
    sampled_dataset: List[str],
    requests_per_rate: int,         # requests
    start_rate: int,                # requests per minute
    end_rate: int,                  # requests per minute
    rate_increase: int              # requests per minute
):
    # for reproducability
    np.random.seed(42)
    curr_rate = start_rate

    while curr_rate < end_rate:
        # requests per minute converted to requests per second
        asyncio.run(send_requests_rate(
            sampled_dataset,
            curr_rate,
            requests_per_rate
        ))
        curr_rate += rate_increase


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Throughput generation script for LLM inference experiments.')
    parser.add_argument(
        '--dataset-path',
        required=True,
        help='The path to the dataset file.'
    )
    parser.add_argument(
        '--num-requests',
        required=True,
        help='The number of requests to sample. Specify 0 or less to sample the entire dataset.'
    )
    args = parser.parse_args()

    sampled_dataset = sample_dataset_prompts(
        args.dataset_path,
        int(args.num_requests)
    )
