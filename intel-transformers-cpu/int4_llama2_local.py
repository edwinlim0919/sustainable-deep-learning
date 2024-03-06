import asyncio
import aiofiles
import time
import argparse
import random
import json
import sys
import numpy as np

from transformers import AutoTokenizer, TextStreamer
from intel_extension_for_transformers.transformers import AutoModelForCausalLM
from concurrent.futures import ProcessPoolExecutor
from functools import partial


result_file_lock = asyncio.Lock()
inference_queue = asyncio.Queue()
result_queue = asyncio.Queue()


model_name = 'TheBloke/Llama-2-7B-Chat-GGUF'
model_file = 'llama-2-7b-chat.Q4_0.gguf'
tokenizer_name = 'meta-llama/Llama-2-7b-chat-hf'

tokenizer = AutoTokenizer.from_pretrained(
    tokenizer_name,
    trust_remote_code=True
)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    model_file=model_file
)

eos_token_id = tokenizer.eos_token_id


def int4_llama2_cpu_inference(prompt: str):
    e2e_inference_start_time = time.time()

    inputs = tokenizer(
        prompt,
        return_tensors='pt'
    ).input_ids

    raw_inference_start_time = time.time()
    outputs = model.generate(
        inputs,
        max_new_tokens=2048,
        eos_token_id=eos_token_id,
        early_stopping=True
    )
    raw_inference_end_time = time.time()
    num_output_tokens = len(outputs[0])

    response = tokenizer.decode(
        outputs[0],
        skip_special_tokens=True
    )

    e2e_inference_end_time = time.time()
    e2e_inference_latency = e2e_inference_end_time - e2e_inference_start_time
    raw_inference_latency = raw_inference_end_time - raw_inference_start_time

    return response, num_output_tokens, e2e_inference_latency, raw_inference_latency


async def async_inference(
    prompt: str,
    executor: ProcessPoolExecutor
) -> str:
    loop = asyncio.get_event_loop()

    response, num_output_tokens, e2e_inference_latency, raw_inference_latency = await loop.run_in_executor(
        executor,
        int4_llama2_cpu_inference,
        prompt
    )

    return response, num_output_tokens, e2e_inference_latency, raw_inference_latency


async def inference_worker(executor: ProcessPoolExecutor):
    while True:
        prompt, curr_rate, requests_per_rate = await inference_queue.get()
        print(f'INFERENCE_WORKER prompt: {prompt}, curr_rate: {curr_rate}, requests_per_rate: {requests_per_rate}')
        sys.stdout.flush()

        response, num_output_tokens, e2e_inference_latency, raw_inference_latency = await async_inference(
            prompt,
            executor
        )

        response_data = {
            'prompt' : prompt,
            'response' : response,
            'num_output_tokens' : num_output_tokens,
            'e2e_inference_latency' : e2e_inference_latency,
            'raw_inference_latency' : raw_inference_latency,
            'curr_rate' : curr_rate,
            'requests_per_rate' : requests_per_rate
        }
        print(f'INFERENCE_WORKER response_data: {response_data}')
        sys.stdout.flush()

        await result_queue.put(response_data)
        inference_queue.task_done()


async def write_results(output_file_path):
    with open(output_file_path, 'a') as file:
        while not result_queue.empty():
            result = await result_queue.get()
            file.write(str(result) + '\n')
            result_queue.task_done()


async def async_main(
    sampled_dataset: list[str],
    requests_per_rate: int,
    start_rate: float,
    end_rate: float,
    increase_rate: float,
    output_file_path: str,
):
    executor = ProcessPoolExecutor()
    worker = asyncio.create_task(inference_worker(executor))

    # for reproducability
    #np.random.seed(42)
    curr_rate = start_rate

    while curr_rate < end_rate:
        print(f'ASYNC_MAIN curr_rate: {curr_rate}')
        sys.stdout.flush()

        lambda_rate = curr_rate / 60
        inter_arrival_times = np.random.exponential(1 / lambda_rate, size=requests_per_rate)
        arrival_times = np.cumsum(inter_arrival_times)

        initial_arrival_time_offset = arrival_times[0] * 0.8
        arrival_times = [arrival_time - initial_arrival_time_offset for arrival_time in arrival_times]
        print(f'ASYNC_MAIN arrival_times: {arrival_times}')
        sys.stdout.flush()

        start_time = time.time()
        tasks = []
        for i in range(requests_per_rate):
            send_time = start_time + arrival_times[i]
            await asyncio.sleep(max(0, send_time - time.time()))
            await inference_queue.put((
                sampled_dataset[i],
                curr_rate,
                requests_per_rate
            ))

        await inference_queue.join()

        # After inferencing is done, write results to output file
        await write_results(output_file_path)
        curr_rate = curr_rate * increase_rate

    worker.cancel()
    executor.shutdown()


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
    print(f'output_file_path: {args.output_file_path}')
    asyncio.run(async_main(
        sampled_dataset,
        args.requests_per_rate,
        args.start_rate,
        args.end_rate,
        args.increase_rate,
        args.output_file_path
    ))
