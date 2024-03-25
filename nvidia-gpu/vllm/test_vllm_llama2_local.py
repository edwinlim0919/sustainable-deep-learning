import asyncio
import argparse

import vllm_llama2_local


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--test-prompt',
        type=str,
        required=True,
        help='test prompt for async engine'
    )
    #parser = AsyncEngineArgs.add_cli_args(parser)
    parser = vllm_llama2_local.add_cli_args_wrapper(parser)
    args = parser.parse_args()

    vllm_llama2_local.vllm_setup(args)
    asyncio.run(vllm_llama2_local.generate(
        args.test_prompt,
        '0'
    ))
