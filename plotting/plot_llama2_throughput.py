import argparse
import ast
import matplotlib.pyplot as plt


# Calculates normalized latency/token for each different request rate
def parse_llama2_throughput(input_file_path: str):
    with open(input_file_path, 'r') as file:
        results_str = file.readlines()

    results_dict = []
    for result_str in results_str:
        results_dict.append(ast.literal_eval(result_str))

    rate_to_lpt_list_e2e = {}
    rate_to_lpt_list_raw = {}
    for result_dict in results_dict:
        num_output_tokens = result_dict['num_output_tokens']
        raw_inference_latency = result_dict['raw_inference_latency']
        e2e_query_time = result_dict['e2e_query_time']
        curr_rate = result_dict['curr_rate']
        print(f'num_output_tokens: {num_output_tokens}, raw_inference_latency: {raw_inference_latency}, e2e_query_time: {e2e_query_time}, curr_rate: {curr_rate}')

        latency_per_token_e2e = e2e_query_time / num_output_tokens
        latency_per_token_raw = raw_inference_latency / num_output_tokens

        if curr_rate not in rate_to_lpt_list_e2e:
            rate_to_lpt_list_e2e[curr_rate] = []
        rate_to_lpt_list_e2e[curr_rate].append(latency_per_token_e2e)

        if curr_rate not in rate_to_lpt_list_raw:
            rate_to_lpt_list_raw[curr_rate] = []
        rate_to_lpt_list_raw[curr_rate].append(latency_per_token_raw)

    print(f'rate_to_lpt_list_e2e: {rate_to_lpt_list_e2e}\n')
    print(f'rate_to_lpt_list_raw: {rate_to_lpt_list_raw}\n')

    rate_to_normalized_latency_e2e = {}
    rate_to_normalized_latency_raw = {}
    for rate, lpt_list in rate_to_lpt_list_e2e.items():
        lpt_list_len = len(lpt_list)
        lpt_list_sum = 0
        for lpt in lpt_list:
            lpt_list_sum += float(lpt)
        normalized_latency = lpt_list_sum / lpt_list_len
        rate_to_normalized_latency_e2e[float(rate)] = normalized_latency

    for rate, lpt_list in rate_to_lpt_list_raw.items():
        lpt_list_len = len(lpt_list)
        lpt_list_sum = 0
        for lpt in lpt_list:
            lpt_list_sum += float(lpt)
        normalized_latency = lpt_list_sum / lpt_list_len
        rate_to_normalized_latency_raw[float(rate)] = normalized_latency

    print(f'rate_to_normalized_latency_e2e: {rate_to_normalized_latency_e2e}\n')
    print(f'rate_to_normalized_latency_raw: {rate_to_normalized_latency_raw}\n')

    return rate_to_normalized_latency_e2e, rate_to_normalized_latency_raw


def plot_llama2_throughput(
    rate_to_normalized_latency_e2e,
    rate_to_normalized_latency_raw,
    output_file_path
):
    sorted_keys_e2e = sorted(rate_to_normalized_latency_e2e.keys())
    sorted_values_e2e = [rate_to_normalized_latency_e2e[key] for key in sorted_keys_e2e]

    plt.figure(figsize=(10, 6))
    plt.plot(sorted_keys_e2e, sorted_values_e2e, marker='o')

    plt.title('Normalized Latency vs. Requests Per Minute')
    plt.xlabel('Requests Per Minute')
    plt.ylabel('Normalized Latency')
    plt.grid(True)

    plt.savefig(output_file_path, format='png')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Plotting script for llama2 throughput experiments.')
    parser.add_argument(
        '--input-file-path',
        required=True,
        type=str,
        help='The path to the input results file.'
    )
    parser.add_argument(
        '--output-file-path',
        required=True,
        type=str,
        help='The path to the output plotting file.'
    )
    args = parser.parse_args()

    rate_to_normalized_latency_e2e, rate_to_normalized_latency_raw = parse_llama2_throughput(args.input_file_path)
    plot_llama2_throughput(
        rate_to_normalized_latency_e2e,
        rate_to_normalized_latency_raw,
        args.output_file_path 
    )
