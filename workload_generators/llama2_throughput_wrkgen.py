import json
import random
import argparse


# Sampling dataset prompts for throughput experiments
def sample_dataset_prompts(
    dataset_path: str,
    num_requests: int
):
    with open(dataset_path, 'r', encoding='utf-8') as f:
        dataset = json.load(f)

    # Filter out the conversations with less than 2 turns.
    dataset = [data for data in dataset if len(data["conversations"]) >= 2]
    # Only keep the first prompt of each conversation.
    dataset = [data["conversations"][0]["value"] for data in dataset]
    if num_requests < 1:
        num_requests = len(dataset)

    print('SAMPLED DATASET')
    sampled_dataset = random.sample(dataset, num_requests)
    print(sampled_dataset)

    return sampled_dataset


#shareGPT_file_path = 'ShareGPT_V3_unfiltered_cleaned_split.json'
#shareGPT_dataset_path = 'ShareGPT_V3_unfiltered_cleaned_split_top100.json'
#sample_dataset_prompts(shareGPT_dataset_path, 0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Throughput generation script for LLM inference experiments.')
    parser.add_argument(
        '--dataset-path',
        required=True,
        help='The path to the dataset file.'
    )
    parser.add_argument(
        '--num-requests',
        required=True,
        help='The number of requests to sample. Specify 0 or less to sample the entire dataset.'
    )
    args = parser.parse_args()

    sampled_dataset = sample_dataset_prompts(
        args.dataset_path,
        int(args.num_requests)
    )
