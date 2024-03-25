import argparse
import asyncio

from typing import Tuple
from vllm.sampling_params import SamplingParams
from vllm.utils import random_uuid
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine


engine = None


async def generate(
    prompt: Tuple[str, int, int],
    sampling_params: SamplingParams
):
    prompt_str = prompt[0]
    num_input_tokens = prompt[1]
    num_output_tokens = prompt[2]
    request_id = random_uuid()

    results_generator = engine.generate(
        prompt_str,
        sampling_params,
        request_id
    )

    # Non-streaming
    final_output = None
    async for request_output in results_generator:
        final_output = request_output

    assert final_output is not None
    prompt = final_output.prompt
    text_outputs = [prompt + output.text for output in final_output.outputs]


async def async_main(
    engine,
    test_prompt
):
    sampling_params = SamplingParams()
    results_generator = engine.generate(
        args.test_prompt,
        sampling_params,
        '0'
    )

    final_output = None
    async for request_output in results_generator:
        final_output = request_output

    print(final_output)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--test-prompt',
        type=str,
        required=True,
        help='test prompt for async engine'
    )
    #parser.add_argument(
    #    '--temperature',
    #    type=float,
    #    required=True,
    #    help='temperature to control randomness of sampling'
    #)
    parser = AsyncEngineArgs.add_cli_args(parser)
    args = parser.parse_args()

    engine_args = AsyncEngineArgs.from_cli_args(args)
    engine = AsyncLLMEngine.from_engine_args(engine_args)

    asyncio.run(async_main(
        engine,
        args.test_prompt
    ))

