# Adapted from TensorRT-LLM/examples/summarize.py for LLM benchmarking
# TODO: Get rid of unused imports

import argparse
import random
import ast
import time
from pathlib import Path

import evaluate
import numpy as np
import torch
from transformers import (AutoModel, AutoModelForCausalLM,
                          AutoModelForSeq2SeqLM, GenerationConfig,
                          AutoTokenizer)
from utils import (DEFAULT_HF_MODEL_DIRS, add_common_args, load_tokenizer,
                   read_model_name)

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
    batch_input_tokens: torch.Tensor,
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
    runtime_rank: int,
    runner: ModelRunnerCpp
):
    batch_start_time = time.time()
    with torch.no_grad():
        batch_outputs = runner.generate(
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
        # TODO: Need to use C++ benchmark to use the in-flight batch manager...
        torch.cuda.synchronize()
    batch_end_time = time.time()

    return batch_outputs, batch_start_time, batch_end_time


def main(args):
    runtime_rank = tensorrt_llm.mpi_rank()
    logger.set_level(args.log_level)
    model_name, model_version = read_model_name(args.engine_dir)
    logger.info(f'MAIN model_name: {model_name}')
    logger.info(f'MAIN model_version: {model_version}')

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
    num_iterations = args.num_iterations

    # Setting random seeds
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)

    # Loading tokenizer
    profiler.start('MAIN load tokenizer')
    tokenizer, pad_id, end_id = benchmark_utils.load_tokenizer(
        args.tokenizer_dir,
        args.add_special_tokens
    )
    profiler.stop('MAIN load tokenizer')
    logger.info(f'MAIN load tokenizer takes: {profiler.elapsed_time_in_sec("load tokenizer")} sec')

    # Sampling the dataset
    if args.use_prompt_formatting:
        sampled_prompts = benchmark_utils.sample_dataset_prompts_with_formatting(
            args.dataset_path,
            args.num_requests_sample,
            max_output_tokens,
            max_input_tokens,
            tokenizer
        )
    else:
        sampled_prompts = benchmark_utils.sample_dataset_prompts_no_formatting(
            args.dataset_path,
            args.num_requests_sample,
            max_output_tokens,
            max_input_tokens,
            tokenizer
        )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    container_output_dir = Path(args.container_output_dir)
    container_output_dir.mkdir(exist_ok=True, parents=True)

    if runtime_rank == 0:
        logger.info(f'MAIN creating output directory {args.output_dir} w/ files {args.output_file}')
        with (output_dir / args.output_file).open('w') as f:
            f.write(f'engine path: {args.engine_dir}\n')
            f.write(f'tokenizer path: {args.tokenizer_dir}\n')

    # Check if stop file has been uploaded to the container
    with (container_output_dir / args.container_stop_file).open('r') as f:
        container_stop_lines = f.readlines()
    assert('RUNNING' in container_stop_lines[0])

    if not PYTHON_BINDINGS:
        logger.warning("MAIN Python bindings of C++ session is unavailable, fallback to Python session.")
    runner_cls = ModelRunnerCpp

    runner_kwargs = dict(engine_dir=args.engine_dir,
                         rank=runtime_rank,
                         debug_mode=args.debug_mode,
                         gpu_weights_percent=args.gpu_weights_percent)
    if args.medusa_choices is not None:
        args.medusa_choices = ast.literal_eval(args.medusa_choices)
        assert args.temperature == 1.0, "Medusa should use temperature == 1.0"
        assert args.num_beams == 1, "Medusa should use num_beams == 1"
        runner_kwargs.update(medusa_choices=args.medusa_choices)
    if not args.use_py_session:
        runner_kwargs.update(
            max_batch_size=max_batch_size,
            max_input_len=max_input_tokens,
            max_output_len=max_output_tokens,
            max_beam_width=num_beams,
            max_attention_window_size=max_attention_window_size,
            sink_token_length=sink_token_length,
            max_tokens_in_paged_kv_cache=args.max_tokens_in_paged_kv_cache,
            kv_cache_enable_block_reuse=args.kv_cache_enable_block_reuse,
            kv_cache_free_gpu_memory_fraction=args.kv_cache_free_gpu_memory_fraction,
            enable_chunked_context=args.enable_chunked_context
        )
    runner = runner_cls.from_dir(**runner_kwargs)

    # set max_output_tokens and max_input_tokens to reflect current batch results from GPT4
    # decouple batch sampling with actual runtime
    batch_dicts = []
    for iteration in range(num_iterations):
        batch_inputs = random.sample(sampled_prompts, max_batch_size)
        curr_max_input_tokens = 0
        curr_max_output_tokens = 0
        batch_input_prompts = []

        for batch_input in batch_inputs:
            if batch_input[1] > curr_max_input_tokens:
                curr_max_input_tokens = batch_input[1]
            if batch_input[2] > curr_max_output_tokens:
                curr_max_output_tokens = batch_input[2]
            batch_input_prompts.append(batch_input[0])

        batch_input_tokens = benchmark_utils.prepare_inputs(
            batch_input_prompts,
            args.add_special_tokens,
            tokenizer,
            curr_max_input_tokens
        )
        batch_input_lengths = [x.size(0) for x in batch_input_tokens]

        batch_dict = {
            'iteration': iteration,
            'batch_size': max_batch_size,
            'max_input_tokens': curr_max_input_tokens,
            'max_output_tokens': curr_max_output_tokens,
            'batch_input_prompts': batch_input_prompts,
            'batch_input_tokens': batch_input_tokens,
            'batch_input_lengths': batch_input_lengths
        }
        batch_dicts.append(batch_dict)

    for iteration in range(num_iterations):
        if runtime_rank == 0:
            logger.info(f'MAIN iteration: {iteration} / {num_iterations - 1}')

        batch_dict = batch_dicts[iteration]
        batch_outputs, batch_start_time, batch_end_time = eval_trt_llm(
            batch_dict['batch_input_tokens'],
            batch_dict['max_input_tokens'],
            batch_dict['max_output_tokens'],
            max_attention_window_size,
            sink_token_length,
            end_id,
            pad_id,
            temperature,
            top_k,
            top_p,
            num_beams,
            length_penalty,
            early_stopping,
            repetition_penalty,
            presence_penalty,
            frequency_penalty,
            runtime_rank,
            runner
        )
        batch_dict['batch_outputs'] = batch_outputs
        batch_dict['batch_start_time'] = batch_start_time
        batch_dict['batch_end_time'] = batch_end_time

    # parsing and writing results
    if runtime_rank == 0:
        with (output_dir / args.output_file).open('a') as file:
            file.write(f'num_iterations: {num_iterations}\n')
            for batch_dict in batch_dicts:
                benchmark_utils.parse_batch_dict(
                    batch_dict,
                    tokenizer
                )
                benchmark_utils.write_batch_dict(
                    batch_dict,
                    file,
                    args.no_token_logging
                )

    # Sleep for 30s for extra nvsmi readings
    time.sleep(30)
    if runtime_rank == 0:
        with (container_output_dir / args.container_stop_file).open('w') as f:
            f.write('COMPLETED\n')
    del runner


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
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
        '--num_iterations',
        type=int,
        required=True,
        help='number of batch iterations to run during profiling'
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
        '--container_output_dir',
        type=str,
        required=True,
        help='directory in docker container for saving output files'
    )
    parser.add_argument(
        '--container_stop_file',
        type=str,
        required=True,
        help='filepath in docker container for coordinating nvsmi loop stop'
    )
    parser.add_argument(
        '--add_special_tokens',
        default=False,
        action='store_true',
        help='Whether or not to add special tokens'
    )
    parser.add_argument(
        '--use_prompt_formatting',
        default=False,
        action='store_true',
        help='Whether or not to use prompt formatting for better generation.'
    )
    parser.add_argument(
        '--no_token_logging',
        default=False,
        action='store_true',
        help='Specify this argument to avoid absurdly long output logs by not saving the token output.'
    )
    parser = add_common_args(parser)
    args = parser.parse_args()
    main(args)
