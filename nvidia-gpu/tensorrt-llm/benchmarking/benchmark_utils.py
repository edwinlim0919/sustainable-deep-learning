import asyncio
import time
import argparse
import random
import json
import sys
import numpy as np

from transformers import AutoTokenizer


def load_tokenizer(
    tokenizer_dir,
):
    if 'llama' in tokenizer_dir:
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_dir,
            legacy=False,
            padding_side='left',
            truncation_side='left',
            trust_remote_code=True,
            use_fast=True
        )
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
        pad_id = tokenizer.pad_token_id
        end_id = tokenizer.eos_token_id
        return tokenizer, pad_id, end_id


def prepare_inputs(batch_input_texts: list[str],
                   add_special_tokens: bool,
                   tokenizer: AutoTokenizer,
                   max_input_tokens: int
):
    batch_input_ids = tokenizer(
        batch_input_texts,
        return_tensors='pt',
        add_special_tokens=add_special_tokens,
        truncation=True,
        padding=True,
        max_length=max_input_tokens
    ).input_ids
    return batch_input_ids


def parse_batch_dict(
    batch_dict: dict,
    tokenizer: AutoTokenizer
):
    batch_outputs = batch_dict['batch_outputs']
    del batch_dict['batch_outputs']

    batch_output_ids = batch_outputs['output_ids']
    batch_sequence_lengths = batch_outputs['sequence_lengths']
    device = batch_output_ids.device
    dtype = batch_output_ids.dtype
    batch_output_tokens = []
    for t in batch_output_ids:
        batch_output_tokens.append(t.squeeze().tolist())
    batch_output_completions = tokenizer.batch_decode(batch_output_tokens)

    batch_dict['batch_output_completions'] = batch_output_completions
    batch_dict['batch_input_tokens'] = batch_dict['batch_input_tokens'].tolist()
    batch_dict['batch_output_tokens'] = batch_output_tokens
    batch_dict['batch_output_lengths'] = batch_sequence_lengths.tolist()
    batch_dict['device'] = device
    batch_dict['dtype'] = dtype

    for key, value in batch_dict.items():
        print(f'parse_batch_dict key: {key}, value: {value}')
    print()


#def write_results(
#
#):


# General Llama2 prompt formatting given a list of message dicts
# Prompt interleaving should look like: <human> <gpt> <human> <gpt> ...
# Adapted from code in https://huggingface.co/TheBloke/Llama-2-13B-chat-GPTQ/discussions/5
# TODO: Prompting like this seems to over-moralize the model...
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
    max_output_tokens: int,
    max_input_tokens: int,
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

    prompts = [data[0]['content'] for data in dataset]
    prompt_token_ids = tokenizer(prompts).input_ids
    completions = [data[1]['content'] for data in dataset]
    completion_token_ids = tokenizer(completions).input_ids

    # Filter out too long or too short sequences
    assert(len(prompts) == len(completions))
    assert(len(prompt_token_ids) == len(completion_token_ids))
    filtered_dataset = []
    for i in range(len(prompts)):
        num_prompt_tokens = len(prompt_token_ids[i])
        num_completion_tokens = len(completion_token_ids[i])
        if num_prompt_tokens < 4 or num_completion_tokens < 4:
            continue
        if num_prompt_tokens > max_input_tokens or num_completion_tokens > max_output_tokens:
            continue
        filtered_dataset.append((
            prompts[i],
            num_prompt_tokens,
            num_completion_tokens
        ))

    # Sample the prompts
    if num_requests_sample < 1:
        num_requests_sample = len(filtered_dataset)
    sampled_prompts = random.sample(filtered_dataset, num_requests_sample)

    return sampled_prompts
