import argparse
import ast
import matplotlib.pyplot as plt
import numpy as np
import io
import contextlib

from calflops import calculate_flops, calculate_flops_hf
from transformers import LlamaTokenizer, LlamaForCausalLM


def parse_calflops_stdout(calflops_output):
    calflops_lines = calflops_output.split('\n')
    calflops_section, detailed_calflops_section = False, False
    calflops_info, detailed_calflops_info = {}, {}

    for line in calflops_lines:
        stripped = line.strip()
        stripped_split = stripped.split()
        
        if 'Calculate Flops Results' in stripped:
            calflops_section = True
        if 'Detailed Calculated FLOPs Results' in stripped:
            calflops_section = False
            detailed_calflops_section = True

        if calflops_section:
            # Parse data in calfops section
            # TODO: This data should be consistently formatted across different models, but not 100% sure
            if 'Total Training Params: ' in stripped:
                calflops_info['total_training_params'] = stripped_split[-2]
                calflops_info['total_training_params_units'] = stripped_split[-1]
            if 'fwd MACs: ' in stripped:
                calflops_info['fwd_MACs'] = stripped_split[-2]
                calflops_info['fwd_MACs_units'] = stripped_split[-1]
            if 'fwd FLOPs: ' in stripped:
                calflops_info['fwd_FLOPs'] = stripped_split[-2]
                calflops_info['fwd_FLOPs_units'] = stripped_split[-1]
            if 'fwd+bwd MACs: ' in stripped:
                calflops_info['fwd_bwd_MACs'] = stripped_split[-2]
                calflops_info['fwd_bwd_MACs_units'] = stripped_split[-1]
            if 'fwd+bwd FLOPs: ' in stripped:
                calflops_info['fwd_bwd_FLOPs'] = stripped_split[-2]
                calflops_info['fwd_bwd_FLOPs_units'] = stripped_split[-1]

        print(line)

        #if detailed_calflops_section:
            #print(f'DETAILED_CALFLOPS_SECTION: {line}')
            # This data is model-specific, so need custom parsing


    #for key, val in calflops_info.items():
    #    print(f'CALFLOPS_INFO {key}: {val}')



# TODO: not great since we have to download model weights to run calflops locally, but only thing that works ATM
def plot_model_flops_scaling_local(
    sequence_lengths: list[int],
    model_names: list[str],
    plot_name: str,
    plot_filename: str
):
    for model_name in model_names:
        model = LlamaForCausalLM.from_pretrained(model_name)
        tokenizer = LlamaTokenizer.from_pretrained(model_name)

        for sequence_length in sequence_lengths:
            # capture stdout of calculate_flops function
            stdout_capture = io.StringIO()

            with contextlib.redirect_stdout(stdout_capture):
                flops, macs, params = calculate_flops(
                    model=model,
                    input_shape=(1, sequence_length), # for FLOPs scaling, just consider single sequence
                    transformer_tokenizer=tokenizer
                )

            output = stdout_capture.getvalue()
            parse_calflops_stdout(output)
            #print(f'PRINTED: {output}')
            #print(f'RETURNED: {model_name} {sequence_length}: {flops}, {macs}, {params}')


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
        plot_model_flops_scaling_local(
            sequence_lengths,
            model_names,
            'Inference FLOPs Scaling',
            'llm_inference_flops_scaling.png'
        )
    elif args.generate_plot == 'calflops_dev_1':
        # compare calflops empirical results from calculations based on "Scaling Laws for Neural Language Models"
        sequence_lengths = [256]
        model_names = ['meta-llama/Llama-2-7b-chat-hf']
        plot_model_flops_scaling_local(
            sequence_lengths,
            model_names,
            'Calflops Dev 1',
            'calflops_dev_1.png'
        )
    elif args.generate_plot == 'calflops_dev_2':
        sequence_lengths = [256]
        model_names = [
            #'meta-llama/Llama-2-7b-chat-hf',
            #'meta-llama/Llama-2-13b-chat-hf',
            'meta-llama/Llama-2-70b-chat-hf'
        ]
        plot_model_flops_scaling_local(
            sequence_lengths,
            model_names,
            'Calflops Dev 2',
            'calflops_dev_2.png'
        )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--generate_plot',
        type=str,
        required=True,
        help='specify the name of the plot to generate'
    )
    parser.add_argument(
        '--hf_access_token',
        type=str,
        required=True,
        help='specify huggingface access token'
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
