import torch
from transformers import BertModel, BertTokenizer


class BertModel:
    def __init__(self):
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    def tokenize(self, text: str) -> str:
        query_ids = self.tokenizer.encode(text, return_tensors='pt')
        #query_tokens = self.tokenizer.convert_ids_to_tokens(query_ids)
        #query_ids = torch.LongTensor(query_ids)
        print('query_ids: ' + str(query_ids))
        print('type of query_ids' + str(type(query_ids)))
        #print('query_tokens: ' + str(query_tokens))
        return 'e'


bertmodel = BertModel()
bertmodel.tokenize('granola bars')
