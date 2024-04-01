# Adapted from TensorRT-LLM/examples/summarize.py for LLM benchmarking
# TODO: Get rid of unused imports

import argparse
import ast
from pathlib import Path

import evaluate
import numpy as np
import torch
from transformers import (AutoModel, AutoModelForCausalLM,
                          AutoModelForSeq2SeqLM, GenerationConfig)
from utils import DEFAULT_HF_MODEL_DIRS, load_tokenizer, read_model_name

import tensorrt_llm
import tensorrt_llm.profiler as profiler
from tensorrt_llm._utils import str_dtype_to_torch
from tensorrt_llm.logger import logger
from tensorrt_llm.models.qwen.utils import make_context
from tensorrt_llm.runtime import PYTHON_BINDINGS, ModelRunner
from tensorrt_llm.tools.ppl import ppl

import benchmark_utils

if PYTHON_BINDINGS:
    from tensorrt_llm.runtime import ModelRunnerCpp


def main(args):
    runtime_rank = tensorrt_llm.mpi_rank()
    logger.set_level(args.log_level)
    model_name, model_version = read_model_name(args.engine_dir)
    logger.info(f'benchmark_trtllm main model_name: {model_name}')
    logger.info(f'benchmark_trtllm main model_version: {model_version}')

    profiler.start('load tokenizer')
    tokenizer, pad_id, end_id = load_tokenizer(
        tokenizer_dir=args.tokenizer_dir,
        vocab_file=None,
        model_name=model_name,
        model_version=model_version
    )
    profiler.stop('load tokenizer')
    logger.info(f'Load tokenizer takes: {profiler.elapsed_time_in_sec("load tokenizer")} sec')

    sampled_prompts = benchmark_utils.sample_dataset_prompts(
        args.dataset_path,
        args.num_requests_sample,
        tokenizer
    )
    sampled_prompts_len = len(sampled_prompts)
    logger.info(f'benchmark_trtllm main sampled_prompts_len: {sampled_prompts_len}')
    for sampled_prompt in sampled_prompts:
        logger.info(f'benchmark_trtllm main sampled_prompt: {sampled_prompt}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--tokenizer_dir',
        type=str,
        required=True,
        help='tokenizer path'
    )
    parser.add_argument(
        '--engine_dir',
        type=str,
        required=True,
        help='trtllm engine path'
    )
    parser.add_argument(
        '--dataset_path',
        type=str,
        required=True,
        help='dataset path'
    )
    parser.add_argument(
        '--num_requests_sample',
        type=int,
        required=True,
        help='number of prompts to sample from dataset'
    )
    parser.add_argument(
        '--max_batch_size',
        type=int,
        required=True,
        help='maximum input batch size'
    )
    parser.add_argument(
        '--max_input_tokens',
        type=int,
        required=True,
        help='maximum input prompt length'
    )
    parser.add_argument(
        '--max_output_tokens',
        type=int,
        required=True,
        help='maximum output generation length'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        required=True,
        help='directory where to save outputs'
    )
    parser.add_argument(
        '--max_attention_window_size',
        type=int,
        default=None,
        help='attention window size for sliding window attention / cyclic kv cache behavior'
    )
    parser.add_argument(
        '--sink_token_length',
        type=int,
        default=None,
        help='sink token length.')
    parser.add_argument('--num_beams', type=int, default=1)
    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument('--top_k', type=int, default=1)
    parser.add_argument('--top_p', type=float, default=0.0)
    parser.add_argument('--length_penalty', type=float, default=1.0)
    parser.add_argument('--repetition_penalty', type=float, default=1.0)
    parser.add_argument('--presence_penalty', type=float, default=0.0)
    parser.add_argument('--frequency_penalty', type=float, default=0.0)
    parser.add_argument(
        '--early_stopping',
        type=int,
        default=1,
        help='use early stopping if num_beams > 1'
        '1 for early-stopping, 0 for non-early-stopping'
        'other values for stopping by length'
    )
    parser.add_argument(
        '--debug_mode',
        default=False,
        action='store_true',
        help='whether or not to turn on the debug mode'
    )
    parser.add_argument(
        '--add_special_tokens',
        default=False,
        action='store_true',
        help='Whether or not to add special tokens'
    )
    parser.add_argument('--log_level', type=str, default='info')
    args = parser.parse_args()

    print('Hello Docker! (fuck you)')
    main(args)
