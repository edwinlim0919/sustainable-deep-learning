import torch
from transformers import AutoTokenizer, RobertaModel, RobertaForQuestionAnswering


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


# Roberta for question answering
class RobertaQAInferenceRay:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained('deepset/roberta-base-squad2')
        self.model = RobertaForQuestionAnswering.from_pretrained("deepset/roberta-base-squad2")

    # Answers a question given an additional text for context
    def inference(self, question: str, text: str) -> str:
        inputs = self.tokenizer(question, text, return_tensors='pt')
        with torch.no_grad():
            outputs = self.model(**inputs)

        answer_start_index = outputs.start_logits.argmax()
        answer_end_index = outputs.end_logits.argmax()
        predict_answer_tokens = inputs.input_ids[0, answer_start_index : answer_end_index + 1]
        return self.tokenizer.decode(predict_answer_tokens, skip_special_tokens=True)


roberta_base_model = RobertaBaseInferenceRay()
cl4_inf_result = roberta_base_model.inference('granola bars')
print('cl4_inf_result: ' + cl4_inf_result)

roberta_qa_model = RobertaQAInferenceRay()
qa_inf_result = roberta_qa_model.inference('Who was Jim Henson?', 'Jim Henson was a nice puppet')
print(qa_inf_result)
