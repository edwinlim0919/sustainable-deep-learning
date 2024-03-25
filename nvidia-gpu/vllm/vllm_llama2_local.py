import time

from typing import Tuple
from vllm.sampling_params import SamplingParams
from vllm.utils import random_uuid
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.engine.arg_utils import AsyncEngineArgs


vllm_engine = None


def add_cli_args_wrapper(parser):
    return AsyncEngineArgs.add_cli_args(parser)


def vllm_setup(args):
    global vllm_engine
    engine_args = AsyncEngineArgs.from_cli_args(args)
    vllm_engine = AsyncLLMEngine.from_engine_args(engine_args)


async def generate(
    test_prompt,
    request_id
):
    sampling_params = SamplingParams()

    request_enqueue_time = time.time()
    results_generator = vllm_engine.generate(
        test_prompt,
        sampling_params,
        request_id
    )

    final_output = None
    async for request_output in results_generator:
        final_output = request_output
    request_dequeue_time = time.time()

    request_latency = request_dequeue_time - request_enqueue_time
    print(f'final_output: {final_output}')
    print(f'request_latency: {request_latency}')
