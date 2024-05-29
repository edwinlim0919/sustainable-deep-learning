import argparse
import matplotlib.pyplot as plt


all_model_info = {
    'meta-llama/Llama-2-7b-chat-hf': {
        'd_model'       : 4096,  # hidden_size
        'd_attn'        : 4096,  # hidden_size
        'd_embd'        : 4096,  # hidden_size
        'd_ff'          : 11008, # intermediate_size
        'n_layer'       : 32,    # num_hidden_layers
        'n_vocab'       : 32000, # vocab_size
        'model_size_GB' : 13.48,
        'attn_comp'     : 'MHA'
    },
    'meta-llama/Llama-2-13b-chat-hf': {
        'd_model'       : 5120,  # hidden_size
        'd_attn'        : 5120,  # hidden_size
        'd_embd'        : 5120,  # hidden_size
        'd_ff'          : 13824, # intermediate_size
        'n_layer'       : 40,    # num_hidden_layers
        'n_vocab'       : 32000, # vocab_size
        'model_size_GB' : 26.03,
        'attn_comp'     : 'MHA'
    }
}


# calculations based on "Scaling Laws for Neural Language Models"
def calculate_model_flops(
    model_name: str,
    input_sequence_length: int,
    output_sequence_length: int,
    use_kv_cache: bool
):
    assert(output_sequence_length >= input_sequence_length)

    model_info = all_model_info[model_name]
    d_model = model_info['d_model']
    d_attn = model_info['d_attn']
    d_embd = model_info['d_embd']
    d_ff = model_info['d_ff']
    n_layer = model_info['n_layer']
    n_vocab = model_info['n_vocab']
    attn_comp = model_info['attn_comp']
    assert(attn_comp == 'MHA' or
           attn_comp == 'GQA')

    total_sequence_flops = 0
    per_token_flops_list = []
    for i in range(output_sequence_length - input_sequence_length):
        # For each token, calculate the number of FLOPs for each stage of inference
        # Context includes original input sequence and any generate output tokens
        n_ctx = input_sequence_length + i

        # Multi-Head Attention
        if attn_comp == 'MHA':
            # Positional embeddings (TODO: or rotary embeddings? Zhihao said it should be negligible)
            embedding_flops = 4 * d_model

            if use_kv_cache:
                # w/ kv-cache, only need to compute q tensor
                attention_qkv_flops = 2 * n_layer * d_model * d_attn
            else:
                # w/o kv-cache, compute k, q, v tensors for each token
                attention_qkv_flops = 2 * n_layer * d_model * 3 * d_attn

            # Masking and final attention calculation still required
            attention_mask_flops = 2 * n_layer * n_ctx * d_attn
            attention_project_flops = 2 * n_layer * d_attn * d_embd

            # Feedforward
            feedforward_flops = 2 * n_layer * 2 * d_model * d_ff
            # De-embed
            deembed_flops = 2 * d_model * n_vocab

        # Grouped-Query Attention
        elif attn_comp == 'GQA':
            # Same as MHA
            embedding_flops = 4 * d_model

            if use_kv_cache:
                # TODO
                # w/ kv-cache...
            else:
                # w/o kv-cache, compute k, q, v tensors for each token
                attention_qkv_flops =


        per_token_flops = embedding_flops + \
                          attention_qkv_flops + \
                          attention_mask_flops + \
                          attention_project_flops + \
                          feedforward_flops + \
                          deembed_flops
        total_sequence_flops += per_token_flops
        per_token_flops_list.append((n_ctx, per_token_flops))

        #print(f'n_ctx: {n_ctx}, per_token_flops: {per_token_flops}')
    return total_sequence_flops, per_token_flops_list


def flops_scaling(
    model_names: list[str],
    input_sequence_lengths: list[int],
    output_sequence_lengths: list[int],
    use_kv_cache: bool
):
    assert(len(input_sequence_lengths) == len(output_sequence_lengths))

    for model_name in model_names:
        for i in range(len(input_sequence_lengths)):
            input_sequence_length = input_sequence_lengths[i]
            output_sequence_length = output_sequence_lengths[i]
            total_sequence_flops, per_token_flops_list = calculate_model_flops(
                model_name,
                input_sequence_length,
                output_sequence_length,
                use_kv_cache
            )
            print(f'{model_name} {input_sequence_length} {output_sequence_length}: {total_sequence_flops}')


# estimate inference memory requirements based on size of kv-cache (if used), weights, and activations
def calculate_model_memory(
    model_name: str,
    input_sequence_length: int,
    output_sequence_length: int,
    use_kv_cache: bool
):
    # TODO: add support for GQA (currently calculates memory for MHA)
    assert(output_sequence_length >= input_sequence_length)

    model_info = all_model_info[model_name]
    d_model = model_info['d_model']
    d_attn = model_info['d_attn']
    d_embd = model_info['d_embd']
    d_ff = model_info['d_ff']
    n_layer = model_info['n_layer']
    n_vocab = model_info['n_vocab']
    model_size_GB = model_info['model_size_GB']


def main(args):
    if args.generate_plot == 'theoretical_dev_0':
        model_names = ['meta-llama/Llama-2-7b-chat-hf']
        input_sequence_lengths = [0]
        output_sequence_lengths = [256]
        use_kv_cache = True
        flops_scaling(
            model_names,
            input_sequence_lengths,
            output_sequence_lengths,
            use_kv_cache
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
