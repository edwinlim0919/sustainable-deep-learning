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
    runtime_rank: int,
    runner: ModelRunnerCpp
):
    batch_size = len(batch_input_prompts)
    batch_input_tokens = benchmark_utils.prepare_inputs(
        batch_input_prompts,
        add_special_tokens,
        tokenizer,
        max_input_tokens
    )
    batch_input_lengths = [x.size(0) for x in batch_input_tokens]

    logger.info(f'EVAL_TRT_LLM batch_size: {batch_size}')
    logger.info(f'EVAL_TRT_LLM batch_input_tokens: {batch_input_tokens}')
    logger.info(f'EVAL_TRT_LLM batch_input_lengths {batch_input_lengths}')

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
        # TODO: Need to use C++ benchmark to use the in-flight batch manager...
        torch.cuda.synchronize()

    logger.info(f'EVAL_TRT_LLM outputs: {outputs}')
    output_ids = outputs['output_ids']
    output_sequence_lengths = outputs['sequence_lengths']
    decoded_outputs = tokenizer.batch_decode(output_ids)
    logger.info(f'EVAL_TRT_LLM decoded_outputs: {decoded_outputs}')
    logger.info(f'EVAL_TRT_LLM output_sequence_lengths: {output_sequence_lengths}')

    ## Extract a list of tensors of shape beam_width x output_ids.
    #if runtime_rank == 0:
    #    output_ids = outputs['output_ids']
    #    output_beams_list = [
    #        tokenizer.batch_decode(output_ids[batch_idx, :,
    #                                          input_lengths[batch_idx]:],
    #                               skip_special_tokens=True)
    #        for batch_idx in range(batch_size)
    #    ]
    #    output_ids_list = [
    #        output_ids[batch_idx, :, input_lengths[batch_idx]:]
    #        for batch_idx in range(batch_size)
    #    ]

    #    ppls = [[] for _ in range(batch_size)]
    #    seq_lengths_array = outputs["sequence_lengths"].cpu().tolist()
    #    lengths_info = {
    #        'input_lengths': input_lengths,
    #        'seq_lengths': seq_lengths_array
    #    }

    #    return output_beams_list, output_ids_list, ppls, lengths_info
    #return [], [], [], {}


def main(args):
    runtime_rank = tensorrt_llm.mpi_rank()
    logger.set_level(args.log_level)
    model_name, model_version = read_model_name(args.engine_dir)
    logger.info(f'benchmark_trtllm main model_name: {model_name}')
    logger.info(f'benchmark_trtllm main model_version: {model_version}')

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
        max_output_tokens,
        max_input_tokens,
        tokenizer
    )
    sampled_prompts_len = len(sampled_prompts)
    batch_input_prompts = [sampled_prompt[0] for sampled_prompt in sampled_prompts]
    logger.info(f'benchmark_trtllm main sampled_prompts_len: {sampled_prompts_len}')

    logger.info(f'Creating output directory {args.output_dir} w/ file {args.output_file}')
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    with (output_dir / args.output_file).open('w') as f:
        f.write(f'Engine path: {args.engine_dir}\n')
        f.write(f'Tokenizer path: {args.tokenizer_dir}\n')

    # TODO: Add random_seed flag in gptj
    #metric_tensorrt_llm = [evaluate.load("rouge") for _ in range(num_beams)]
    #for i in range(num_beams):
    #    metric_tensorrt_llm[i].seed = random_seed
    #ppls_trt_llm = [[] for _ in range(num_beams)]

    #sampled_prompt_tokens = benchmark_utils.prepare_inputs(
    #    sampled_prompts_text_only,
    #    args.add_special_tokens,
    #    tokenizer,
    #    max_input_tokens
    #)
    #logger.info(f'sampled_prompt_tokens: {sampled_prompt_tokens}')

    if not PYTHON_BINDINGS:
        logger.warning(
            "Python bindings of C++ session is unavailable, fallback to Python session."
        )
    runner_cls = ModelRunnerCpp
    runner_kwargs = dict(engine_dir=args.engine_dir,
                         rank=runtime_rank,
                         debug_mode=args.debug_mode)
    runner_kwargs.update(
        max_batch_size=max_batch_size,
        max_input_len=max_input_tokens,
        max_output_len=max_output_tokens,
        max_beam_width=num_beams,
        max_attention_window_size=max_attention_window_size,
        sink_token_length=sink_token_length)
    runner = runner_cls.from_dir(**runner_kwargs)

    # TODO: set max_output_tokens to the max output tokens of this batch
    eval_trt_llm(
        batch_input_prompts,
        args.add_special_tokens,
        tokenizer,
        max_input_tokens,
        max_output_tokens,
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

    #datapoint = dataset[0:1]
    #output, *_ = eval_trt_llm(datapoint,
    #                          eval_task=args.eval_task,
    #                          eval_ppl=args.eval_ppl,
    #                          add_special_tokens=args.add_special_tokens)
    #if runtime_rank == 0:
    #    logger.info(
    #        "---------------------------------------------------------")
    #    logger.info("TensorRT-LLM Generated : ")
    #    logger.info(f" Input : {datapoint[dataset_input_key]}")
    #    logger.info(f"\n Reference : {datapoint[dataset_output_key]}")
    #    logger.info(f"\n Output : {output}")
    #    logger.info(
    #        "---------------------------------------------------------")

    #ite_count = 0
    #data_point_idx = 0
    #total_output_token_count_trt_llm = 0  # only valid for runtime_rank == 0
    #while (data_point_idx < len(dataset)) and (ite_count < args.max_ite):
    #    if runtime_rank == 0:
    #        logger.debug(
    #            f"run data_point {data_point_idx} ~ {data_point_idx + max_batch_size}"
    #        )
    #    datapoint = dataset[data_point_idx:(data_point_idx +
    #                                        max_batch_size)]

    #    profiler.start('tensorrt_llm')
    #    output_tensorrt_llm, output_ids_trt_llm, curr_ppls_trt_llm, lengths_info = eval_trt_llm(
    #        datapoint,
    #        eval_task=args.eval_task,
    #        eval_ppl=args.eval_ppl,
    #        add_special_tokens=args.add_special_tokens)
    #    profiler.stop('tensorrt_llm')
    #    if runtime_rank == 0:
    #        input_lengths = lengths_info['input_lengths']
    #        seq_lengths = lengths_info['seq_lengths']
    #        output_token_count_trt_llm = sum(
    #            seq_lengths[idx][0] - input_lengths[idx]
    #            for idx in range(len(input_lengths)))
    #        total_output_token_count_trt_llm += output_token_count_trt_llm

    #    if runtime_rank == 0:
    #        for batch_idx in range(len(output_tensorrt_llm)):
    #            for beam_idx in range(num_beams):
    #                metric_tensorrt_llm[beam_idx].add_batch(
    #                    predictions=[
    #                        output_tensorrt_llm[batch_idx][beam_idx]
    #                    ],
    #                    references=[
    #                        datapoint[dataset_output_key][batch_idx]
    #                    ])
    #                if args.eval_ppl:
    #                    ppls_trt_llm[beam_idx].append(
    #                        curr_ppls_trt_llm[batch_idx][beam_idx])
    #        if output_dir is not None:
    #            for i in range(len(output_tensorrt_llm[0])):
    #                for beam_idx in range(num_beams):
    #                    with (output_dir / 'trtllm.out').open('a') as f:
    #                        f.write(
    #                            f'[{data_point_idx + i}] [Beam {beam_idx}] {output_tensorrt_llm[beam_idx][i]}\n'
    #                        )

    #        logger.debug('-' * 100)
    #        logger.debug(f"Input : {datapoint[dataset_input_key]}")
    #        logger.debug(f'TensorRT-LLM Output: {output_tensorrt_llm}')
    #        logger.debug(f"Reference : {datapoint[dataset_output_key]}")

    #    data_point_idx += max_batch_size
    #    ite_count += 1
    del runner



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
