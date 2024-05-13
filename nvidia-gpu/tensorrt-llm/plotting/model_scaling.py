import argparse
import ast
import matplotlib.pyplot as plt

import numpy as np
from calflops import calculate_flops_hf


def plot_model_flops_scaling(
    sequence_lengths: list[int],
    model_names = list[str],
    plot_name: str,
    plot_filename: stz
):
    


def main(args):
    # TODO: specify different args for different FLOPs
    if args.generate_plot == 'flops_scaling_gpt2_llama2_128_4096':
        sequence_lengths = [
            128,
            256,
            512,
            1024,
            2048,
            4096
        ]
        model_names = [
            'openai-community/gpt2',
            'openai-community/gpt2-medium',
            'openai-community/gpt2-large',
            'openai-community/gpt2-xl',
            'meta-llama/Llama-2-7b-chat-hf',
            'meta-llama/Llama-2-13b-chat-hf',
            'meta-llama/Llama-2-70b-chat-hf'
        ]
        plot_model_flops_scaling(
            sequence_lengths,
            model_names,
            'Inference FLOPs Scaling',
            'llm_inference_flops_scaling.png'
        )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--generate_plot',
        type=str,
        required=True,
        help='specify the name of the plot to generate'
    )
    args = parser.parse_args()
    main(args)

#llama2_model_params = {
#    '7B': {
#        'd_model': 4096,
#        'n_layer': 32,
#        'd_attn': 4096,
#        'd_ff': 4096
#    },
#    '13B': {
#        'd_model': 5120,
#        'n_layer': 40,
#        'd_attn': 5120,
#        'd_ff': 5120
#    },
#    '33B': {
#        'd_model': 6656,
#        'n_layer': 60,
#        'd_attn': 6656,
#        'd_ff': 6656
#    },
#    '66B': {
#        'd_model': 8192,
#        'n_layer': 80,
#        'd_attn': 8192,
#        'd_ff': 8192
#    }
#}
#
#
#def calculate_transformer_flops(
#    d_model: int,
#    n_layer: int,
#    d_attn: int,
#    d_ff: int,
#    num_output_tokens: int
#) -> int:
#    N = 2 * d_model * n_layer * (2 * d_attn + d_ff)
#    C_forward = (2 * N + 2 * n_layer * 1 * d_attn) * num_output_tokens
#    return C_forward
#
#
#plot_llama2_tflops = False
#
#if plot_llama2_tflops:
#    sequence_lengths = [128, 256, 512, 1024, 2048, 4096]
#    for model, params in llama2_model_params.items():
#        params['FLOPs'] = {}
#    
#        for sequence_length in sequence_lengths:
#            total_FLOPs = calculate_transformer_flops(
#                params['d_model'],
#                params['n_layer'],
#                params['d_attn'],
#                params['d_ff'],
#                sequence_length
#            )
#            params['FLOPs'][sequence_length] = total_FLOPs
#    
#    for model, params in llama2_model_params.items():
#        print(f'{model}: {params}')
#    
#    data = {model: [(details['FLOPs'][length] / (10 ** 12)) for length in sequence_lengths] for model, details in llama2_model_params.items()}
#    n = len(sequence_lengths)
#    fig, ax = plt.subplots()
#    index = np.arange(n)
#    bar_width = 0.2
#    opacity = 0.8
#    
#    for i, (model, flops) in enumerate(data.items()):
#        ax.bar(index + i * bar_width, flops, bar_width, alpha=opacity, label=model)
#    
#    ax.set_xlabel('Sequence Length')
#    ax.set_ylabel('TFLOPs')
#    ax.set_title('TFLOPs of Llama2 Models by Sequence Length')
#    ax.set_xticks(index + bar_width)
#    ax.set_xticklabels(sequence_lengths)
#    ax.legend()
#    
#    plt.savefig('llama2_flops_scaling.png')
