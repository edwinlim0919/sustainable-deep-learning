import time

from typing import Tuple
from vllm.sampling_params import SamplingParams
from vllm.utils import random_uuid
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.engine.arg_utils import AsyncEngineArgs
from transformers import AutoTokenizer


vllm_engine = None
tokenizer = None


def add_cli_args_wrapper(parser):
    return AsyncEngineArgs.add_cli_args(parser)


def vllm_setup(args):
    global vllm_engine
    global tokenizer

    engine_args = AsyncEngineArgs.from_cli_args(args)
    vllm_engine = AsyncLLMEngine.from_engine_args(engine_args)
    tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-2-7b-chat-hf')


async def generate(
    test_prompt,
    num_output_tokens,
    request_id
):
    prompt_token_ids = tokenizer(
        test_prompt,
        return_tensors='pt'
    ).input_ids
    prompt_token_ids_list = prompt_token_ids.tolist()[0]

    num_input_tokens = len(prompt_token_ids_list)
    max_tokens = min(2000 - num_input_tokens, num_output_tokens)
    sampling_params = SamplingParams(
        max_tokens=max_tokens
    )

    results_generator = vllm_engine.generate(
        test_prompt,
        sampling_params,
        request_id,
        prompt_token_ids_list
    )
    async for request_output in results_generator:
        final_output = request_output

    return final_output
