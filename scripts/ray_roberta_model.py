import torch
from transformers import AutoTokenizer, RobertaModel


# Base model inference for feature vector
class RobertaBaseInferenceRay:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained('roberta-base')
        self.model = RobertaModel.from_pretrained('roberta-base', output_hidden_states=True)

    # Returns a feature vector from mean of concatenated last 4 hidden states
    def inference(self, text: str) -> str:
        query_ids = self.tokenizer.encode(text, return_tensors='pt')
        with torch.no_grad():
            outputs = self.model(input_ids=query_ids)

        hidden_states = outputs[2]
        last_four_layers = [hidden_states[i] for i in (-1, -2, -3, -4)]
        cat_hidden_states = torch.cat(tuple(last_four_layers), dim=-1)
        return str(torch.mean(cat_hidden_states, dim=1).squeeze())





roberta_model = RobertaBaseInferenceRay()
cl4_inf_result = roberta_model.inference('granola bars')
print('cl4_inf_result: ' + cl4_inf_result)
