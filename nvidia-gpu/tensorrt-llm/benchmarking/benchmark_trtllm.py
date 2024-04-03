# Adapted from TensorRT-LLM/examples/summarize.py for LLM benchmarking
# TODO: Get rid of unused imports

import argparse
import ast
from pathlib import Path

import evaluate
import numpy as np
import torch
from transformers import (AutoModel, AutoModelForCausalLM,
                          AutoModelForSeq2SeqLM, GenerationConfig,
                          AutoTokenizer)
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


def eval_trt_llm(
    batch_input_prompts: list[str],
    add_special_tokens: bool,
    tokenizer: AutoTokenizer,
    max_input_tokens: int,
    max_output_tokens: int,
    max_attention_window_size: int | None,
    sink_token_length: int | None,
    end_id: int,
    pad_id: int,
    temperature: float,
    top_k: int,
    top_p: float,
    num_beams: int,
    length_penalty: float,
    early_stopping: int,
    repetition_penalty: float,
    presence_penalty: float,
    frequency_penalty: float,
    
    runner: ModelRunnerCpp
):
    batch_size = len(batch_input_prompts)
    #batch_input_ids = _prepare_inputs(datapoint[dataset_input_key],
    #                                  eval_task=eval_task,
    #                                  add_special_tokens=add_special_tokens)
    batch_input_tokens = benchmark_utils.prepare_inputs(
        batch_input_prompts,
        add_special_tokens,
        tokenizer,
        max_input_tokens
    )
    batch_input_lengths = [x.size(0) for x in batch_input_tokens]

    with torch.no_grad():
        outputs = runner.generate(
            batch_input_tokens,
            max_new_tokens=max_output_tokens,
            max_attention_window_size=max_attention_window_size,
            sink_token_length=sink_token_length,
            end_id=end_id,
            pad_id=pad_id,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            num_beams=num_beams,
            length_penalty=length_penalty,
            early_stopping=early_stopping,
            repetition_penalty=repetition_penalty,
            presence_penalty=presence_penalty,
            frequency_penalty=frequency_penalty,
            output_sequence_lengths=True,
            return_dict=True,
            medusa_choices=None)
        # TODO: Need to use C++ benchmark to use in-flight batch manager...
        torch.cuda.synchronize()

    # Extract a list of tensors of shape beam_width x output_ids.
    if runtime_rank == 0:
        output_ids = outputs['output_ids']
        output_beams_list = [
            tokenizer.batch_decode(output_ids[batch_idx, :,
                                              input_lengths[batch_idx]:],
                                   skip_special_tokens=True)
            for batch_idx in range(batch_size)
        ]
        output_ids_list = [
            output_ids[batch_idx, :, input_lengths[batch_idx]:]
            for batch_idx in range(batch_size)
        ]

        ppls = [[] for _ in range(batch_size)]
        seq_lengths_array = outputs["sequence_lengths"].cpu().tolist()
        lengths_info = {
            'input_lengths': input_lengths,
            'seq_lengths': seq_lengths_array
        }

        return output_beams_list, output_ids_list, ppls, lengths_info
    return [], [], [], {}


def main(args):
    runtime_rank = tensorrt_llm.mpi_rank()
    logger.set_level(args.log_level)
    model_name, model_version = read_model_name(args.engine_dir)
    logger.info(f'benchmark_trtllm main model_name: {model_name}')
    logger.info(f'benchmark_trtllm main model_version: {model_version}')

    # Loading tokenizer
    profiler.start('load tokenizer')
    tokenizer, pad_id, end_id = load_tokenizer(
        tokenizer_dir=args.tokenizer_dir,
        vocab_file=None,
        model_name=model_name,
        model_version=model_version
    )
    profiler.stop('load tokenizer')
    logger.info(f'Load tokenizer takes: {profiler.elapsed_time_in_sec("load tokenizer")} sec')

    # Sampling the dataset
    sampled_prompts = benchmark_utils.sample_dataset_prompts(
        args.dataset_path,
        args.num_requests_sample,
        tokenizer
    )
    sampled_prompts_len = len(sampled_prompts)
    sampled_prompts_text_only = [sampled_prompt[0] for sampled_prompt in sampled_prompts]
    logger.info(f'benchmark_trtllm main sampled_prompts_len: {sampled_prompts_len}')

    # TODO: get rid of this logging
    for sampled_prompt in sampled_prompts:
        logger.info(f'benchmark_trtllm main sampled_prompt: {sampled_prompt}')
    for sampled_prompt in sampled_prompts_text_only:
        logger.info(f'benchmark_trtllm main sampled_prompt: {sampled_prompt}')

    # runtime parameters
    max_batch_size = args.max_batch_size
    top_k = args.top_k
    top_p = args.top_p
    max_output_tokens = args.max_output_tokens
    max_input_tokens = args.max_input_tokens
    max_attention_window_size = args.max_attention_window_size
    sink_token_length = args.sink_token_length

    random_seed = args.random_seed
    temperature = args.temperature
    num_beams = args.num_beams
    length_penalty = args.length_penalty
    early_stopping = args.early_stopping
    repetition_penalty = args.repetition_penalty
    presence_penalty = args.presence_penalty
    frequency_penalty = args.frequency_penalty

    logger.info(f'Creating output directory {args.output_dir} w/ file {args.output_file}')
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    with (output_dir / args.output_file).open('w') as f:
        f.write(f'Engine path: {args.engine_dir}\n')
        f.write(f'Tokenizer path: {args.tokenizer_dir}\n')

    # TODO: Add random_seed flag in gptj
    metric_tensorrt_llm = [evaluate.load("rouge") for _ in range(num_beams)]
    for i in range(num_beams):
        metric_tensorrt_llm[i].seed = random_seed
    ppls_trt_llm = [[] for _ in range(num_beams)]

    sampled_prompt_tokens = benchmark_utils.prepare_inputs(
        sampled_prompts_text_only,
        args.add_special_tokens,
        tokenizer,
        max_input_tokens
    )
    logger.info(f'sampled_prompt_tokens: {sampled_prompt_tokens}')


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
        help='directory for saving output files'
    )
    parser.add_argument(
        '--output_file',
        type=str,
        required=True,
        help='output file name'
    )
    parser.add_argument(
        '--random_seed',
        type=int,
        required=True,
        help='random seed for reproducibility'
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
