import asyncio
import time

from transformers import AutoTokenizer, TextStreamer
from intel_extension_for_transformers.transformers import AutoModelForCausalLM
from concurrent.futures import ProcessPoolExecutor
from functools import partial


model_name = 'TheBloke/Llama-2-7B-Chat-GGUF'
model_file = 'llama-2-7b-chat.Q4_0.gguf'
tokenizer_name = 'meta-llama/Llama-2-7b-chat-hf'

tokenizer = AutoTokenizer.from_pretrained(
    tokenizer_name,
    trust_remote_code=True
)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    model_file=model_file
)
executor = ProcessPoolExecutor()
eos_token_id = self.tokenizer.eos_token_id

    #def tokenize_prompt(self, prompt: str):
    #    inputs = self.tokenizer(
    #        prompt,
    #        return_tensors='pt'
    #    ).input_ids
    #    return inputs

    #def decode_output(self, output):
    #    response = self.tokenizer.decode(
    #        output,
    #        skip_special_tokens=True
    #    )
    #    return response

    #def raw_inference(self, inputs):
    #    raw_inference_start_time = time.time()
    #    outputs = self.model.generate(
    #        inputs,
    #        max_new_tokens=2048,
    #        eos_token_id=self.eos_token_id,
    #        early_stopping=True
    #    )
    #    raw_inference_end_time = time.time()
    #    raw_inference_latency = raw_inference_end_time - raw_inference_start_time

    #    return outputs, raw_inference_latency

    #@staticmethod
    #def prepare_

    async def e2e_inference(self, prompt: str) -> str:
        loop = asyncio.get_event_loop()
        e2e_inference_start_time = time.time()

        #def decode_output():
        #    return self.tokenizer(
        #        prompt,
        #        return_tensors='pt'
        #    ).inputs

        #inputs = await loop.run_in_executor(
        #    self.executor,
        #    decode_output
        #    #partial(
        #    #    self.tokenize_prompt,
        #    #    prompt
        #    #)
        #)

        #outputs, raw_inference_latency = await loop.run_in_executor(
        #    self.executor,
        #    partial(
        #        self.raw_inference,
        #        inputs
        #    )
        #)
        #num_output_tokens = len(outputs[0])

        #response = await loop.run_in_executor(
        #    self.executor,
        #    partial(
        #        self.decode_output,
        #        outputs[0]
        #    )
        #)

        #e2e_inference_end_time = time.time()
        #e2e_inference_latency = e2e_inference_end_time - e2e_inference_start_time
        #return f'{response} {num_output_tokens} {e2e_inference_latency} {raw_inference_latency}'
        #return 'jeff'


#if __name__ == '__main__':
#async def main():
#    llama2_inference = Llama2Int4BaseInferenceRay()
#    inf_test_prompt = 'What are the ingredients of olio de aglio? I do not want the entire recipe, only a list of ingredients.'
#    inf_test_resp = await llama2_inference.e2e_inference(inf_test_prompt)
#    print(f'inf_test_resp: {inf_test_resp}')
#
#
#asyncio.run(main())
