import ray
from ray import serve
from ray.serve.handle import DeploymentHandle
from transformers import AutoTokenizer, TextStreamer
from intel_extension_for_transformers.transformers import AutoModelForCausalLM


@serve.deployment
class Llama2EndpointRay:
    def __init__(self, llama2_int4_base_inference: DeploymentHandle):
        self.llama2_int4_base_inference = llama2_int4_base_inference

    async def __call__(self, http_request):
        #print('Llama2EndpointRay __call__ START')
        request = await http_request.json()
        prompt = request['prompt']
        response = self.llama2_int4_base_inference.inference.remote(prompt)
        #print('Llama2EndpointRay __call__ END')
        return await response


# Base inference setup, no tensor parallelism
@serve.deployment
class Llama2Int4BaseInferenceRay:
    def __init__(self):
        model_name = 'TheBloke/Llama-2-7B-Chat-GGUF'
        model_file = 'llama-2-7b-chat.Q4_0.gguf'
        tokenizer_name = 'meta-llama/Llama-2-7b-chat-hf'

        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name,
            trust_remote_code=True
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            model_file=model_file
        )

    def inference(self, prompt: str) -> str:
        inputs = self.tokenizer(
            prompt,
            return_tensors='pt'
        ).input_ids
        eos_token_id = self.tokenizer.eos_token_id
        outputs = self.model.generate(
            inputs,
            max_new_tokens=1024,
            eos_token_id=eos_token_id,
            early_stopping=True
        )
        response = self.tokenizer.decode(
            outputs[0],
            skip_special_tokens=True
        )
        return response


llama2_int4_base_inference = Llama2Int4BaseInferenceRay.bind()
llama2_endpoint = Llama2EndpointRay.bind(llama2_int4_base_inference)


#llama2_int4_inference = Llama2Int4InferenceRay()
#response = llama2_int4_inference.inference('What are the ingredients of olio de aglio? I do not want the entire recipe, only a list of ingredients.')
#print(response)
