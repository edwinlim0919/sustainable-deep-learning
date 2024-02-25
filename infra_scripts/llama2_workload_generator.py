import json


def sample_requests():
    #shareGPT_file_path = 'ShareGPT_V3_unfiltered_cleaned_split.json'
    shareGPT_file_path = 'ShareGPT_V3_unfiltered_cleaned_split_top100.json'
    with open(shareGPT_file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    
    
    #for convs_entry in data:
    #    convs = convs_entry['conversations']
    #    convs_len = len(convs)
    #    print(f'CONVS_LEN: {convs_len}')
    #    for conv in convs:
    #        if conv['from'] == 'human':
    #            print(conv)
    #    print('\n\n')
