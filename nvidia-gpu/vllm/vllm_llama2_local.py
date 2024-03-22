import argparse

from typing import Tuple
from vllm.sampling_params import SamplingParams
from vllm.utils import random_uuid


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


if __name__ == '__main__':
    # TODO
    parser = argparse.ArgumentParser()

    engine_args =
