import torch
import ray
from ray import serve
from ray.serve.handle import DeploymentHandle
from transformers import AutoTokenizer, RobertaModel, RobertaForQuestionAnswering


@serve.deployment
class RobertaEndpointRay:
    def __init__(self, roberta_base_inference: DeploymentHandle, roberta_qa_inference: DeploymentHandle):
        self.roberta_base_inference = roberta_base_inference
        self.roberta_qa_inference = roberta_qa_inference

    async def __call__(self, http_request):
        request = await http_request.json()
        inference_type = request['inference_type']
        if inference_type == 'RobertaBase':
            inference_text = request['inference_text']
            response = self.roberta_base_inference.inference.remote(inference_text)
        elif inference_type == 'RobertaQA':
            inference_text = request['inference_text']
            inference_question = request['inference_question']
            response = self.roberta_qa_inference.inference.remote(inference_question, inference_text)
        else:
            return 'inference_type not supported'

        return await response


# Base model inference for feature vector
@serve.deployment # TODO: change resource alloc
class RobertaBaseInferenceRay:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained('roberta-base')
        self.model = RobertaModel.from_pretrained('roberta-base', output_hidden_states=True)
        self.model.eval()

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
@serve.deployment # TODO: change resource alloc
class RobertaQAInferenceRay:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained('deepset/roberta-base-squad2')
        self.model = RobertaForQuestionAnswering.from_pretrained("deepset/roberta-base-squad2")
        self.model.eval()

    # Answers a question given an additional text for context
    def inference(self, question: str, text: str) -> str:
        inputs = self.tokenizer(question, text, return_tensors='pt')
        with torch.no_grad():
            outputs = self.model(**inputs)

        answer_start_index = outputs.start_logits.argmax()
        answer_end_index = outputs.end_logits.argmax()
        predict_answer_tokens = inputs.input_ids[0, answer_start_index : answer_end_index + 1]
        return self.tokenizer.decode(predict_answer_tokens, skip_special_tokens=True)


roberta_base_inference = RobertaBaseInferenceRay.bind()
roberta_qa_inference = RobertaQAInferenceRay.bind()
roberta_endpoint = RobertaEndpointRay.bind(roberta_base_inference, roberta_qa_inference)
