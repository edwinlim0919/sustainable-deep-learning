import argparse
import matplotlib.pyplot as plt


all_model_info = {
    'meta-llama/Llama-2-7b-chat-hf': {
        'd_model' : 4096,  # hidden_size
        'd_attn'  : 4096,  # hidden_size
        'd_embd'  : 4096,  # hidden_size
        'd_ff'    : 11008, # intermediate_size
        'n_layer' : 32,    # num_hidden_layers
        'n_vocab' : 32000  # vocab_size
    }
}


# calculations based on "Scaling Laws for Neural Language Models"
def calculate_model_flops(
    model_name: str,
    input_sequence_length: int,
    output_sequence_length: int,
    #use_kv_cache: bool
):
    model_info = all_model_info[model_name]
    d_model = model_info['d_model']
    d_attn = model_info['d_attn']
    d_embd = model_info['d_embd']
    d_ff = model_info['d_ff']
    n_layer = model_info['n_layer']
    n_vocab = model_info['n_vocab']

    for i in range(input_sequence_length - output_sequence_length):
        # Calculate the number of FLOPs for each stage of inference
        embedding_flops = 4 * d_model
        attention_qkv_flops = 2 * n_layer * d_model * 3 * d_attn


#def calculate_model_params(
#    model_name
#):


def main(args):
    if args.generate_plot == 'theoretical_dev_1':
        sequence_lengths = [256]
        model_names = ['meta-llama/Llama-2-7b-chat-hf']
        calculate_model_flops(
            sequence_lengths,
            model_names
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
