import asyncio
import argparse

import vllm_llama2_local


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--test-prompt',
        type=str,
        required=True,
        help='test llama2 prompt for async vllm engine'
    )
    parser.add_argument(
        '--test-output',
        type=str,
        required=True,
        help='test llama2 output for async vllm engine'
    )
    parser = vllm_llama2_local.add_cli_args_wrapper(parser)
    args = parser.parse_args()
    vllm_llama2_local.vllm_setup(args)

    output_token_ids = vllm_llama2_local.tokenizer(
        args.test_output,
        return_tensors='pt'
    ).input_ids[0]
    num_output_tokens = len(output_token_ids)

    print(f'test_prompt: {args.test_prompt}')
    print(f'test_output: {args.test_output}')
    print(f'output_token_ids: {output_token_ids}')
    print(f'num_output_tokens: {num_output_tokens}')
    final_output, request_latency = asyncio.run(vllm_llama2_local.generate(
        args.test_prompt,
        num_output_tokens,
        '0'
    ))
    print(f'final_output: {final_output}')
    print(f'request_latency: {request_latency}')
