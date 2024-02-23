import ray
import time
import datetime
import asyncio
from ray import serve
from ray.serve.handle import DeploymentHandle
from transformers import AutoTokenizer, TextStreamer
from intel_extension_for_transformers.transformers import AutoModelForCausalLM
from asyncio import Queue


@serve.deployment
class Llama2EndpointRay:
    def __init__(self, llama2_int4_base_inference: DeploymentHandle):
        self.llama2_int4_base_inference = llama2_int4_base_inference

    async def __call__(self, http_request):
        request = await http_request.json()
        prompt = request['prompt']
        response = self.llama2_int4_base_inference.inference.remote(prompt)
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
        #start_time = time.time()

        # TODO: pre-tokenize and post-decode?
        #       only inference on this end
        inputs = self.tokenizer(
            prompt,
            return_tensors='pt'
        ).input_ids

        #encode_end = time.time()

        eos_token_id = self.tokenizer.eos_token_id
        outputs = self.model.generate(
            inputs,
            max_new_tokens=1024,
            eos_token_id=eos_token_id,
            early_stopping=True
        )

        #raw_inference_end = time.time()

        num_output_tokens = len(outputs[0])
        response = self.tokenizer.decode(
            outputs[0],
            skip_special_tokens=True
        )

        return response

        #end_time = time.time()

        #encode_time = encode_end - start_time
        #raw_inference_time = raw_inference_end - encode_end
        #decode_time = end_time - raw_inference_end
        #total_inference_time = end_time - start_time

        #raw_seconds_per_token = raw_inference_time / num_output_tokens
        #total_second_per_token = total_inference_time / num_output_tokens

        #encode_time_readable = str(datetime.timedelta(seconds=encode_time))
        #raw_inference_time_readable = str(datetime.timedelta(seconds=raw_inference_time))
        #decode_time_readable = str(datetime.timedelta(seconds=decode_time))
        #total_inference_time_readable = str(datetime.timedelta(seconds=total_inference_time))

        #print(f'ENCODE Time: {encode_time_readable} (hours:min:seconds)')
        #print(f'RAW INFERENCE Time: {raw_inference_time_readable} (hours:min:seconds)')
        #print(f'DECODE Time: {decode_time_readable} (hours:min:seconds)')
        #print(f'TOTAL INFERENCE Time: {total_inference_time_readable} (hours:min:seconds)')

        #elapsed_time_seconds = end_time - start_time
        #seconds_per_token = elapsed_time_seconds / num_output_tokens
        #elapsed_time_readable = str(datetime.timedelta(seconds=elapsed_time_seconds))
        #print(f'INFERENCE Elapsed Time: {elapsed_time_readable} (hours:min:seconds)')
        #print(f'INFERENCE Seconds Per Token: {seconds_per_token}')

        #return response


llama2_int4_base_inference = Llama2Int4BaseInferenceRay.bind()
llama2_endpoint = Llama2EndpointRay.bind(llama2_int4_base_inference)
