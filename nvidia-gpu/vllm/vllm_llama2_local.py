import time

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
    test_prompt,
    num_prompt_tokens,
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


async def inference_loop():
    while True:
        test_prompt, num_prompt_tokens, num_output_tokens = await request_queue.get()
        request_id = random_uuid()
        request_dequeue_time = time.time()

        final_output = await generate(
            test_prompt,
            num_prompt_tokens,
            num_output_tokens,
            request_id
        )

        result_enqueue_time = time.time()
        request_latency = result_enqueue_time - result_dequeue_time
        await result_queue.put((
            final_output,
            request_latency
        ))
