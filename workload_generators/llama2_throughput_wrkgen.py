import json
import random


# Sampling dataset prompts for throughput experiments
def sample_dataset_prompts(dataset_path, num_requests):
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
shareGPT_dataset_path = 'ShareGPT_V3_unfiltered_cleaned_split_top100.json'
sample_dataset_prompts(shareGPT_dataset_path, 0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument(
        '--ssh-list',
        required=True,
        help='The filename containing SSH commands.'
    )
    args = parser.parse_args()
    setup_worker_nodes(args.ssh_list)
