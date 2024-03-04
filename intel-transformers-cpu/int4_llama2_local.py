import asyncio
import time

from transformers import AutoTokenizer, TextStreamer
from intel_extension_for_transformers.transformers import AutoModelForCausalLM
from concurrent.futures import ProcessPoolExecutor


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
        self.executor = ProcessPoolExecutor()
        self.eos_token_id = self.tokenizer.eos_token_id

    def tokenize_inputs(self, prompt: str):
        return self.tokenizer(prompt, return_tensors='pt').input_ids

    def raw_inference(self, inputs):
        return self.model.generate(
            inputs,
            max_new_tokens=2048,
            eos_token_id=self.eos_token_id,
            early_stopping=True
        )

    def e2e_inference(self, prompt: str) -> str:
        e2e_inference_start_time = time.time()
        loop = asyncio.get_event_loop()
        inputs = await loop.run_in_executor(
            self.executor,
            self.tokenize_inputs
        )

        outputs = await loop.run_in_executor(
            self.executor,
            self.raw_inference
        )

        num_output_tokens = str(len(outputs[0]))
        response = self.tokenizer.decode(
            outputs[0],
            skip_special_tokens=True
        )

        e2e_inference_end_time = time.time()
        e2e_inference_latency = e2e_inference_end_time - e2e_inference_start_time

        #return f'{response} {num_output_tokens} {inference_latency}'
