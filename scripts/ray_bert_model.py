import torch
import ray
from ray import serve
from fastapi import FastAPI
from transformers import BertModel, BertTokenizer
from transformers import AutoTokenizer, BertForQuestionAnswering


app = FastAPI()


# Base model inference for feature vector
@serve.deployment(num_replicas=2, ray_actor_options={'num_cpus': 0.2, 'num_gpus': 0})
@serve.ingress(app)
class BertCL4InferenceRay:
    def __init__(self):
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.model = BertModel.from_pretrained('bert-base-uncased', output_hidden_states=True)
        self.model = self.model.to('cpu')
        self.model.eval()

    # Returns a feature vector from mean of concatenated last 4 hidden states
    @app.post('/bert/cl4inference')
    def inference(self, text: str) -> str:
        query_ids = self.tokenizer.encode(text, return_tensors='pt')
        query_ids = query_ids.to('cpu')
        with torch.no_grad():
            out = self.model(input_ids=query_ids)

        hidden_states = out[2]
        last_four_layers = [hidden_states[i] for i in (-1, -2, -3, -4)]
        cat_hidden_states = torch.cat(tuple(last_four_layers), dim=-1)
        return str(torch.mean(cat_hidden_states, dim=1).squeeze())


# BERT for question answering
#class BertQAInferenceRay:
#    def __init__(self):
#        self.tokenizer = AutoTokenizer.from_pretrained("deepset/bert-base-cased-squad2")
#        self.model = BertForQuestionAnswering.from_pretrained("deepset/bert-base-cased-squad2")
#
#    def inference(self, question: str, text: str) -> str:
#        inputs = self.tokenizer(question, text, return_tensors="pt")
#        with torch.no_grad():
#            outputs = self.model(**inputs)
#
#        answer_start_index = outputs.start_logits.argmax()
#        answer_end_index = outputs.end_logits.argmax()
#        predict_answer_tokens = inputs.input_ids[0, answer_start_index : answer_end_index + 1]
#        return self.tokenizer.decode(predict_answer_tokens, skip_special_tokens=True)


bert_inference_app = BertCL4InferenceRay.bind()


#BertCL4Inference = False
#if BertCL4Inference:
#    bertmodel = BertCL4InferenceRay()
#    cat_sentence_embedding = bertmodel.inference('granola bars')
#    print('cat_sentence_embedding: ' + str(cat_sentence_embedding))
#    print('cat_sentence_embedding.size(): ' + str(cat_sentence_embedding.size()))
#
#
#BertQAInference = True
#if BertQAInference:
#    bertmodel = BertQAInferenceRay()
#    answer = bertmodel.inference("Who was Jim Henson?", "Jim Henson was a nice puppet")
#    print(answer)
