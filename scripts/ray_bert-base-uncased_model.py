import torch
from transformers import BertModel, BertTokenizer


class BertModelRayCPU:
    def __init__(self):
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.model = BertModel.from_pretrained('bert-base-uncased', output_hidden_states=True)
        self.model = self.model.to('cpu')
        self.model.eval()

    # Returns a feature vector from mean of concatenated last 4 hidden states
    def CL4_embedding_inference(self, text: str) -> str:
        query_ids = self.tokenizer.encode(text, return_tensors='pt')
        query_ids = query_ids.to('cpu')
        with torch.no_grad():
            out = self.model(input_ids=query_ids)

        hidden_states = out[2]
        last_four_layers = [hidden_states[i] for i in (-1, -2, -3, -4)]
        cat_hidden_states = torch.cat(tuple(last_four_layers), dim=-1)
        return torch.mean(cat_hidden_states, dim=1).squeeze()


bertmodel = BertModelRayCPU()
cat_sentence_embedding = bertmodel.CL4_embedding_inference('granola bars')


print('cat_sentence_embedding: ' + str(cat_sentence_embedding))
print('cat_sentence_embedding.size(): ' + str(cat_sentence_embedding.size()))

