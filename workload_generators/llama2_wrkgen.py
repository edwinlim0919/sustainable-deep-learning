import asyncio
import time
import argparse
import random
import json
import sys
import numpy as np

from transformers import AutoTokenizer
from concurrent.futures import ProcessPoolExecutor

sys.path.append('/dev/shm/sustainable-deep-learning/nvidia-gpu/vllm')
import vllm_llama2_local


# Llama2 prompting
B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"
DEFAULT_SYSTEM_PROMPT = f"""You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Please ensure that your responses are socially unbiased and positive in nature. If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."""


async def write_results(
    output_file_path,
    result_queue
):
    # Make sure all the tasks are done
    print(f'write_results {output_file_path}')
    sys.stdout.flush()
    with open(output_file_path, 'a') as file:
        while not result_queue.empty():
            result = await result_queue.get()
            file.write(str(result) + '\n')
            result_queue.task_done()


# Request generation for seconds_per_rate seconds for each rate
async def async_main(
    sampled_prompts: list[str],
    sampled_prompts_len: int,
    seconds_per_rate: int,
    rate_list: list[float],
    output_file_path: str,
    request_queue: asyncio.Queue,
    result_queue: asyncio.Queue
):
    executor = ProcessPoolExecutor()
    worker = asyncio.create_task(vllm_llama2_local.inference_loop(executor))
    #worker = asyncio.create_task(vllm_llama2_local.inference_loop())

    for curr_rate in rate_list:
        print(f'ASYNC_MAIN curr_rate: {curr_rate}')
        sys.stdout.flush()

        lambda_rate = curr_rate / 60
        expected_arrivals = int(lambda_rate * seconds_per_rate)
        inter_arrival_times = np.random.exponential(1 / lambda_rate, size=expected_arrivals)
        arrival_times = np.cumsum(inter_arrival_times)
        print(f'ASYNC_MAIN arrival_times: {arrival_times}')
        sys.stdout.flush()

        curr_rate_start_time = time.time()
        curr_rate_time_limit = curr_rate_start_time + seconds_per_rate
        for i in range(len(arrival_times)):
            send_time = curr_rate_start_time + arrival_times[i]
            sampled_prompt = sampled_prompts[i % sampled_prompts_len]
            await asyncio.sleep(max(0, send_time - time.time()))
            await request_queue.put((
                sampled_prompt,
                curr_rate,
                seconds_per_rate,
                curr_rate_time_limit
            ))

        await request_queue.join()
        # After inferencing is done, write results to output file
        await write_results(
            output_file_path,
            result_queue
        )

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
    assert (all([prompt['role'] == 'human' for prompt in prompts[::2]]) and
            all([prompt['role'] == 'gpt' for prompt in prompts[1::2]]) and
            len(prompts) % 2 == 0)
    prompts_list = [
        f'{B_INST} {(human["content"]).strip()} {E_INST} {(gpt["content"]).strip()}'
        for human, gpt in zip(prompts[::2], prompts[1::2])
    ]
    prompts_list[-1] = prompts_list[-1] + f' {B_INST}'

    return "".join(prompts_list)


# Sampling dataset prompts for throughput experiments
def sample_dataset_prompts(
    dataset_path: str,
    num_requests_sample: int,
    tokenizer: AutoTokenizer
):
    with open(dataset_path, 'r', encoding='utf-8') as f:
        dataset = json.load(f)

    # Filter out the conversations with less than 2 turns
    dataset = [data for data in dataset if len(data["conversations"]) >= 2]

    # Only keep conversations that were initiated by a human and responded by GPT
    human_initiated_dataset = []
    for data in dataset:
        if (data['conversations'][0]['from'] == 'human' and
            data['conversations'][1]['from'] == 'gpt'):
            human_initiated_dataset.append(data)
    dataset = human_initiated_dataset

    # Only keep the first two turns of each conversation and use Llama2 dict format
    llama2_dict_dataset = []
    for data in dataset:
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

    # Format with Llama2 prompt style
    llama2_format_dataset = []
    for data in dataset:
        llama2_conv = llama2_prompt_general(data).split(E_INST)
        llama2_human = f'{llama2_conv[0]} {E_INST}'
        llama2_gpt = f'{E_INST} {llama2_conv[1]}'
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
    prompts = [data[0]['content'] for data in dataset]
    prompt_token_ids = tokenizer(prompts).input_ids
    completions = [data[1]['content'] for data in dataset]
    completion_token_ids = tokenizer(completions).input_ids

    # Filter out too long or too short sequences
    # Real limit is 1020 for prompts, but doing 1000 just to make sure
    assert(len(dataset) == len(prompts) and
           len(dataset) == len(completions))
    filtered_dataset = []
    for i in range(len(dataset)):
        num_prompt_tokens = len(prompt_token_ids[i])
        num_completion_tokens = len(completion_token_ids[i])
        if num_prompt_tokens < 4 or num_completion_tokens < 4:
            continue
        if num_prompt_tokens > 1000 or num_prompt_tokens + num_completion_tokens > 2000:
            continue
        filtered_dataset.append(dataset[i])
    dataset = filtered_dataset

    # Get prompt, prompt tokens, and # output length
    llama2_prompts = []
    for data in dataset:
        llama2_human = data[0]['content']
        llama2_gpt = data[1]['content']
        llama2_human_tokens = tokenizer(llama2_human).input_ids
        llama2_gpt_tokens = tokenizer(llama2_gpt).input_ids
        llama2_prompts.append((
            llama2_human,
            len(llama2_human_tokens),
            len(llama2_gpt_tokens)
        ))

    # Sample the prompts
    if num_requests_sample < 1:
        num_requests_sample = len(llama2_prompts)
    sampled_prompts = random.sample(llama2_prompts, num_requests_sample)

    return sampled_prompts


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='throughput generation script for LLM inference experiments')
    parser.add_argument(
        '--dataset-path',
        required=True,
        type=str,
        help='path to the dataset file'
    )
    parser.add_argument(
        '--output-file-path',
        required=True,
        type=str,
        help='path to the output file'
    )
    parser.add_argument(
        '--num-requests-sample',
        required=True,
        type=int,
        help='number of requests to sample. Specify 0 or less to sample the entire dataset'
    )
    parser.add_argument(
        '--seconds-per-rate',
        required=True,
        type=int,
        help='number of seconds to send per request rate'
    )
    parser.add_argument(
        '--start-rate',
        required=False,
        type=float,
        help='starting request rate in requests per minute'
    )
    parser.add_argument(
        '--end-rate',
        required=False,
        type=float,
        help='ending request rate in requests per minute'
    )
    parser.add_argument(
        '--increase-rate',
        required=False,
        type=float,
        help='request rate multiplicative increase per iteration'
    )
    parser.add_argument(
        '--rate-list',
        required=False,
        type=float,
        nargs='+',
        help='a list of request rates for the experiment'
    )
    parser.add_argument(
        '--wrkgen-seed',
        required=True,
        type=int,
        help='random seed for experiment reproducibility'
    )

    # TODO Need to change this to incorporate other frameworks
    parser = vllm_llama2_local.add_cli_args_wrapper(parser)
    args = parser.parse_args()
    vllm_llama2_local.vllm_setup(args)

    # Rates should be specified as list or geometric increase
    if not ((args.start_rate and args.end_rate and args.increase_rate) or
            args.rate_list):
        raise ValueError('Rates must be specified as a list or a geometric increase')

    # Generate request rate list for geometric increase
    if not args.rate_list:
        rate_list = []
        curr_rate = args.start_rate
        while curr_rate <= args.end_rate:
            rate_list.append(curr_rate)
            curr_rate *= args.increase_rate
    else:
        rate_list = args.rate_list

    # Set randomness seeds for reproducibility
    random.seed(args.wrkgen_seed)
    np.random.seed(args.wrkgen_seed)

    # TODO: Need to change this to incorporate other frameworks
    tokenizer = vllm_llama2_local.tokenizer
    request_queue = vllm_llama2_local.request_queue
    result_queue = vllm_llama2_local.result_queue

    # Throughput experiment
    print(f'Sampling dataset {args.dataset_path}...')
    sampled_prompts = sample_dataset_prompts(
        args.dataset_path,
        args.num_requests_sample,
        tokenizer
    )
    sampled_prompts_len = len(sampled_prompts)

    print('Generating requests...')
    print(f'sampled_prompts_len: {sampled_prompts_len}')
    print(f'start_rate: {args.start_rate}')
    print(f'end_rate: {args.end_rate}')
    print(f'increase_rate: {args.increase_rate}')
    print(f'output_file_path: {args.output_file_path}')
    print(f'seconds_per_rate: {args.seconds_per_rate}')
    asyncio.run(async_main(
        sampled_prompts,
        sampled_prompts_len,
        args.seconds_per_rate,
        rate_list,
        args.output_file_path,
        request_queue,
        result_queue
    ))
    print('Done.')
