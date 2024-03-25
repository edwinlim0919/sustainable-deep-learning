import time
import sys

from vllm.sampling_params import SamplingParams
from vllm.utils import random_uuid
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.utils import random_uuid
from transformers import AutoTokenizer


# vLLM engine and Llama2 tokenizer
vllm_engine = None
tokenizer = None

# Queuing for asynchronous request generation
request_queue = None
result_queue = None


def add_cli_args_wrapper(parser):
    return AsyncEngineArgs.add_cli_args(parser)


def vllm_setup(args):
    global vllm_engine
    global tokenizer
    global request_queue
    global result_queue

    engine_args = AsyncEngineArgs.from_cli_args(args)
    vllm_engine = AsyncLLMEngine.from_engine_args(engine_args)
    tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-2-7b-chat-hf')
    request_queue = asyncio.Queue()
    result_queue = asyncio.Queue()


async def generate(
    prompt,
    num_prompt_tokens,
    num_output_tokens,
    request_id
):
    prompt_token_ids = tokenizer(
        prompt,
        return_tensors='pt'
    ).input_ids
    prompt_token_ids_list = prompt_token_ids.tolist()[0]

    max_tokens = min(2000 - num_prompt_tokens, num_output_tokens)
    sampling_params = SamplingParams(
        max_tokens=max_tokens
    )

    results_generator = vllm_engine.generate(
        prompt,
        sampling_params,
        request_id,
        prompt_token_ids_list
    )
    async for request_output in results_generator:
        final_output = request_output

    return final_output


async def inference_loop():
    while True:
        prompt, curr_rate, seconds_per_rate, time_limit = await request_queue.get()
        print(f'INFERENCE_LOOP prompt: {prompt}, curr_rate: {curr_rate}, seconds_per_rate: {seconds_per_rate}, time_limit: {time_limit}')
        sys.stdout.flush()
        request_id = random_uuid()
        request_dequeue_time = time.time()

        if request_dequeue_time > time_limit:
            print('INFERENCE_LOOP TIME LIMIT EXCEEDED')
            sys.stdout.flush()
            request_queue.task_done()
            continue

        final_output = await generate(
            prompt[0],
            prompt[1],
            prompt[2],
            request_id
        )
        result_enqueue_time = time.time()
        request_latency = result_enqueue_time - result_dequeue_time
        response_data = {
            'final_output': final_output,
            'request_dequeue_time': request_dequeue_time,
            'result_enqueue_time': result_enqueue_time,
            'request_latency': request_latency,
            'curr_rate': curr_rate,
            'seconds_per_rate': seconds_per_rate,
            'time_limit': time_limit
        }

        await result_queue.put(response_data)
        request_queue.task_done()
