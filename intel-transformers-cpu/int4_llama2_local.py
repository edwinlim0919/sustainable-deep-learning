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


# Explicit queuing
result_file_lock = asyncio.Lock()
inference_queue = asyncio.Queue()
result_queue = asyncio.Queue()


# Llama model stuff
model_name = 'TheBloke/Llama-2-7B-Chat-GGUF'
model_file = 'llama-2-7b-chat.Q4_0.gguf'
tokenizer_name = 'meta-llama/Llama-2-7b-chat-hf'

B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"
DEFAULT_SYSTEM_PROMPT = f"""You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Please ensure that your responses are socially unbiased and positive in nature. If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."""

tokenizer = AutoTokenizer.from_pretrained(
    tokenizer_name,
    trust_remote_code=True
)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    model_file=model_file
)


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
        early_stopping=True,
        repetition_penalty=1.1
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
        prompt, curr_rate, requests_per_rate, seconds_per_rate, inference_enqueue_time, time_limit = await inference_queue.get()
        print(f'INFERENCE_WORKER prompt: {prompt}, curr_rate: {curr_rate}, requests_per_rate: {requests_per_rate}, seconds_per_rate: {seconds_per_rate}')
        sys.stdout.flush()

        if time.time() > time_limit:
            inference_queue.task_done()
            continue

        response, num_output_tokens, e2e_inference_latency, raw_inference_latency = await async_inference(
            prompt,
            executor
        )
        result_enqueue_time = time.time()
        e2e_query_time = result_enqueue_time - inference_enqueue_time

        response_data = {
            'prompt': prompt,
            'response': response,
            'num_output_tokens': num_output_tokens,
            'e2e_inference_latency': e2e_inference_latency,
            'raw_inference_latency': raw_inference_latency,
            'e2e_query_time': e2e_query_time,
            'curr_rate': curr_rate,
            'requests_per_rate': requests_per_rate,
            'seconds_per_rate': seconds_per_rate
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


# Request generation for reqests_per_rate requests for each rate
async def async_main_requests(
    sampled_prompts: list[str],
    requests_per_rate: int,
    start_rate: float,
    end_rate: float,
    increase_rate: float,
    output_file_path: str,
):
    executor = ProcessPoolExecutor()
    worker = asyncio.create_task(inference_worker(executor))
    curr_rate = start_rate

    while curr_rate <= end_rate:
        print(f'ASYNC_MAIN_REQUESTS curr_rate: {curr_rate}')
        sys.stdout.flush()

        lambda_rate = curr_rate / 60
        inter_arrival_times = np.random.exponential(1 / lambda_rate, size=requests_per_rate)
        arrival_times = np.cumsum(inter_arrival_times)
        print(f'ASYNC_MAIN_REQUESTS arrival_times: {arrival_times}')
        sys.stdout.flush()

        start_time = time.time()
        time_limit = start_time + sys.maxsize
        for i in range(requests_per_rate):
            send_time = start_time + arrival_times[i]
            await asyncio.sleep(max(0, send_time - time.time()))
            inference_enqueue_time = time.time()
            await inference_queue.put((
                sampled_prompts[i],
                curr_rate,
                requests_per_rate,
                -1,
                inference_enqueue_time,
                time_limit
            ))

        await inference_queue.join()

        # After inferencing is done, write results to output file
        await write_results(output_file_path)
        curr_rate = curr_rate * increase_rate

    worker.cancel()
    executor.shutdown()


# Request generation for seconds_per_rate seconds for each rate
async def async_main_seconds(
    sampled_prompts: list[str],
    seconds_per_rate: int,
    start_rate: float,
    end_rate: float,
    increase_rate: float,
    output_file_path: str,
):
    executor = ProcessPoolExecutor()
    worker = asyncio.create_task(inference_worker(executor))
    curr_rate = start_rate

    while curr_rate <= end_rate:
        print(f'ASYNC_MAIN_TIME curr_rate: {curr_rate}')
        sys.stdout.flush()

        lambda_rate = curr_rate / 60
        expected_arrivals = int(lambda_rate * seconds_per_rate)
        inter_arrival_times = np.random.exponential(1 / lambda_rate, size=expected_arrivals)
        arrival_times = np.cumsum(inter_arrival_times)
        print(f'ASYNC_MAIN_TIME arrival_times: {arrival_times}')
        sys.stdout.flush()

        start_time = time.time()
        time_limit = start_time + seconds_per_rate
        sampled_prompts_len = len(sampled_prompts)
        for i in range(len(arrival_times)):
            send_time = start_time + arrival_times[i]
            sampled_prompt = sampled_prompts[i % sampled_prompts_len]
            await asyncio.sleep(max(0, send_time - time.time()))
            inference_enqueue_time = time.time()
            await inference_queue.put((
                sampled_prompt,
                curr_rate,
                -1,
                seconds_per_rate,
                inference_enqueue_time,
                time_limit
            ))

        await inference_queue.join()

        # After inferencing is done, write results to output file
        await write_results(output_file_path)
        curr_rate = curr_rate * increase_rate

    worker.cancel()
    executor.shutdown()


# General Llama2 prompt formatting given a list of message dicts
# Prompt interleaving should look like: <human> <gpt> <human> <gpt> ...
# Adapted from code in https://huggingface.co/TheBloke/Llama-2-13B-chat-GPTQ/discussions/5
def llama2_prompt_general(prompts: list[dict]):
    B_INST, E_INST = "[INST]", "[/INST]"
    B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"
    DEFAULT_SYSTEM_PROMPT = f"""You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Please ensure that your responses are socially unbiased and positive in nature. If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."""

    if prompts[0]["role"] != "system":
        prompts = [{
            "role": "system",
            "content": DEFAULT_SYSTEM_PROMPT
        }] + prompts
    prompts = [{
        "role": prompts[1]["role"],
        "content": B_SYS + prompts[0]["content"] + E_SYS + prompts[1]["content"],
    }] + prompts[2:]

    # Ensure that user prompts first, and there is a gpt response for every human query
    #print(f'PROMPTS: {prompts}\n\n')
    assert (all([prompt['role'] == 'human' for prompt in prompts[::2]]) and
            all([prompt['role'] == 'gpt' for prompt in prompts[1::2]]) and
            len(prompts) % 2 == 0)
    prompts_list = [
        f'{B_INST} {(human["content"]).strip()} {E_INST} {(gpt["content"]).strip()}'
        for human, gpt in zip(prompts[::2], prompts[1::2])
    ]
    prompts_list[-1] = prompts_list[-1] + f' {B_INST}'

    return "".join(prompts_list)


# Simple Llama2 prompt formatting given a single human prompt
def llama2_prompt_single(prompt: str):
    return f'[INST] <<SYS>>\n{DEFAULT_SYSTEM_PROMPT}\n<</SYS>>\n\n{prompt} [/INST]'


# Sampling dataset prompts for throughput experiments
def sample_dataset_prompts(
    dataset_path: str,
    num_requests_sample: int
):
    with open(dataset_path, 'r', encoding='utf-8') as f:
        dataset = json.load(f)

    # Filter out the conversations with less than 2 turns
    dataset = [data for data in dataset if len(data["conversations"]) >= 2]

    # Only keep conversations that were initiated by a human
    human_initiated_dataset = []
    for data in dataset:
        if data['conversations'][0]['from'] == 'human':
            human_initiated_dataset.append(data)
    dataset = human_initiated_dataset

    # Only keep the first two turns of each conversation and use Llama2 dict format
    llama2_dict_dataset = []
    for data in dataset:
        if (data['conversations'][0]['from'] != 'human' or
            data['conversations'][1]['from'] != 'gpt'):
            continue

        human_dict = {
            'role': data['conversations'][0]['from'],
            'content': data['conversations'][0]['value']
        }
        gpt_dict = {
            'role': data['conversations'][1]['from'],
            'content': data['conversations'][1]['value']
        }
        llama2_dict_dataset.append([
            human_dict,
            gpt_dict
        ])
    dataset = llama2_dict_dataset

    # TODO: Test this out before running more throughput tests
    # Format with Llama2 prompt style
    llama2_format_dataset = []
    for data in dataset:
        llama2_conv = llama2_prompt_general(data).split(E_INST)
        llama2_human = llama2_conv[0] + f' {E_INST}'
        llama2_gpt = f'{E_INST} ' + llama2_conv[1]

        human_dict = {
            'role': data[0]['role'],
            'content': llama2_human
        }
        gpt_dict = {
            'role': data[1]['role'],
            'content': llama2_gpt
        }
        llama2_format_dataset.append([
            human_dict,
            gpt_dict
        ])
    dataset = llama2_format_dataset

    # Tokenize the prompts and completions
    prompts = [prompt[0]['content'] for prompt in dataset]
    prompt_token_ids = tokenizer(prompts).input_ids
    completions = [prompt[1]['content'] for prompt in dataset]
    completion_token_ids = tokenizer(completions).input_ids

    # Filter out too long or too short sequences
    assert(len(dataset) == len(prompts) and
           len(dataset) == len(completions))
    filtered_dataset = []
    for i in range(len(dataset)):
        num_prompt_tokens = len(prompt_token_ids[i])
        num_completion_tokens = len(completion_token_ids[i])
        if num_prompt_tokens < 4 or num_completion_tokens < 4:
            continue
        if num_prompt_tokens > 1020 or num_prompt_tokens + num_completion_tokens > 2040:
            continue
        filtered_dataset.append(dataset[i])
    dataset = filtered_dataset

    # Augment conversation turns with Llama2 prompt format
    # Currently only uses human turn
    #llama2_prompts = []
    #for data in dataset:
    #    llama2_conv = llama2_prompt_general(data).split(E_INST)
    #    llama2_prompt = llama2_conv[0] + f' {E_INST}'
    #    llama2_prompts.append(llama2_prompt)

    # Get human turns
    llama2_prompts = []
    for data in dataset:
        llama2_human = data[0]['content']
        llama2_prompts.append(llama2_human)

    # Sample the prompts
    if num_requests_sample < 1:
        num_requests_sample = len(llama2_prompts)
    sampled_prompts = random.sample(llama2_prompts, num_requests_sample)

    return sampled_prompts


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Throughput generation script for LLM inference experiments.')

    # Throughput generation experiment arguments
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
        help='The path to the output file.'
    )
    parser.add_argument(
        '--num-requests-sample',
        required=True,
        type=int,
        help='The number of requests to sample. Specify 0 or less to sample the entire dataset.'
    )
    parser.add_argument(
        '--requests-per-rate',
        required=False,
        type=int,
        help='The number of requests to send per request rate.'
    )
    parser.add_argument(
        '--seconds-per-rate',
        required=False,
        type=int,
        help='The number of seconds to send per request rate.'
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

    # Misc. testing arguments
    parser.add_argument(
        '--prompt',
        required=False,
        type=str,
        help='Provide a specifc prompt to test on the model.'
    )
    args = parser.parse_args()

    if args.prompt:
        prompt = llama2_prompt_single(args.prompt)
        print(f'Test request: {prompt}')
        response, num_output_tokens, e2e_inference_latency, raw_inference_latency = int4_llama2_cpu_inference(prompt)
        print(f'response: {response}')
        print(f'num_output_tokens {num_output_tokens}')
        print(f'e2e_inference_latency {e2e_inference_latency}')
        print(f'raw_inference_latency {raw_inference_latency}')
    else:
        if (not args.requests_per_rate and
            not args.seconds_per_rate):
            raise ValueError('Need to specify either --requests-per-rate or --seconds-per-rate')

        print(f'Sampling dataset {args.dataset_path}...')
        sampled_prompts = sample_dataset_prompts(
            args.dataset_path,
            args.num_requests_sample
        )
        sampled_prompts_len = len(sampled_prompts)

        print('Generating requests...')
        print(f'sampled_prompts_len: {sampled_prompts_len}')
        print(f'start_rate: {args.start_rate}')
        print(f'end_rate: {args.end_rate}')
        print(f'increase_rate: {args.increase_rate}')
        print(f'output_file_path: {args.output_file_path}')
        if args.requests_per_rate:
            print(f'requests_per_rate: {args.requests_per_rate}')
            asyncio.run(async_main_requests(
                sampled_prompts,
                args.requests_per_rate,
                args.start_rate,
                args.end_rate,
                args.increase_rate,
                args.output_file_path
            ))
        else: # seconds_per_rate
            print(f'seconds_per_rate: {args.seconds_per_rate}')
            asyncio.run(async_main_seconds(
                sampled_prompts,
                args.seconds_per_rate,
                args.start_rate,
                args.end_rate,
                args.increase_rate,
                args.output_file_path
            ))
