import json


def sample_requests(dataset_path):
    with open(dataset_path, 'r', encoding='utf-8') as f:
        dataset = json.load(f)
    # Filter out the conversations with less than 2 turns.
    dataset = [data for data in dataset if len(data["conversations"]) >= 2]
    # Only keep the first two turns of each conversation.
    dataset = [(data["conversations"][0]["value"],
                data["conversations"][1]["value"]) for data in dataset]

    print(dataset)

    #for convs_entry in data:
    #    convs = convs_entry['conversations']
    #    convs_len = len(convs)
    #    print(f'CONVS_LEN: {convs_len}')
    #    for conv in convs:
    #        if conv['from'] == 'human':
    #            print(conv)
    #    print('\n\n')



#shareGPT_file_path = 'ShareGPT_V3_unfiltered_cleaned_split.json'
shareGPT_dataset_path = 'ShareGPT_V3_unfiltered_cleaned_split_top100.json'
sample_requests(shareGPT_dataset_path)
