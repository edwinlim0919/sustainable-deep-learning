import torch
import ray
from ray import serve
from ray.serve.handle import DeploymentHandle
from fastapi import FastAPI
from transformers import BertModel, BertTokenizer
from transformers import AutoTokenizer, BertForQuestionAnswering


@serve.deployment
class BertEndpointRay:
    def __init__(self, bert_cl4_inference: DeploymentHandle, bert_qa_inference: DeploymentHandle):
        self.bert_cl4_inference = bert_cl4_inference
        self.bert_qa_inference = bert_qa_inference

    async def __call__(self, http_request):
        request = await http_request.json()
        inference_type = request['inference_type']
        if inference_type == 'BertCL4':
            inference_text = request['inference_text']
            response = self.bert_cl4_inference.inference.remote(inference_text)
        elif inference_type == 'BertQA':
            inference_text = request['inference_text']
            inference_question = request['inference_question']
            response = self.bert_qa_inference.inference.remote(inference_question, inference_text)
        else:
            return 'inference_type not supported'

        return await response


# Base model inference for feature vector
#@serve.deployment(num_replicas=2, ray_actor_options={'num_cpus': 0.2, 'num_gpus': 0}) # TODO: change resource alloc
@serve.deployment
class BertCL4InferenceRay:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        self.model = BertModel.from_pretrained('bert-base-uncased', output_hidden_states=True)
        self.model = self.model.to('cpu')
        self.model.eval()

    # Returns a feature vector from mean of concatenated last 4 hidden states
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
#@serve.deployment(num_replicas=2, ray_actor_options={'num_cpus': 0.2, 'num_gpus': 0}) # TODO: change resource alloc
@serve.deployment
class BertQAInferenceRay:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("deepset/bert-base-cased-squad2")
        self.model = BertForQuestionAnswering.from_pretrained("deepset/bert-base-cased-squad2")

    # Answers a question given an additional text for context
    def inference(self, question: str, text: str) -> str:
        inputs = self.tokenizer(question, text, return_tensors="pt")
        with torch.no_grad():
            outputs = self.model(**inputs)

        answer_start_index = outputs.start_logits.argmax()
        answer_end_index = outputs.end_logits.argmax()
        predict_answer_tokens = inputs.input_ids[0, answer_start_index : answer_end_index + 1]
        return self.tokenizer.decode(predict_answer_tokens, skip_special_tokens=True)


bert_cl4_inference = BertCL4InferenceRay.bind()
bert_qa_inference = BertQAInferenceRay.bind()
bert_endpoint = BertEndpointRay.bind(bert_cl4_inference, bert_qa_inference)
