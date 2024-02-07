import torch
import transformers
from torch import cuda, bfloat16
from transformers import LlamaForCausalLM, LlamaTokenizer
from transformers import StoppingCriteria, StoppingCriteriaList


#class Llama7BChatEndpointRay:


# TODO: Should make this optional, can potentially quantify sustainability benefits of quantization
# TODO: Quantization is only available for GPUs
# Enabling quanitzation for large model with less memory requirements
bnb_config = transformers.BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type='nf4',
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=bfloat16
)


# Custom stopping criteria object
class StopOnTokens(StoppingCriteria):
    def __init__(self, stop_token_ids):
        self.stop_token_ids = stop_token_ids

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        for stop_ids in self.stop_token_ids:
            if torch.eq(input_ids[0][-len(stop_ids):], stop_ids).all():
                return True
        return False


class Llama7BChatQAInferenceRay:
    def __init__(self):
        model_dir = '../model_weights/llama-2-7b-chat-hf'
        device = f'cuda:{cuda.current_device()}' if cuda.is_available() else 'cpu'
        self.model_config = transformers.AutoConfig.from_pretrained(model_dir)
        self.model = transformers.AutoModelForCausalLM.from_pretrained(
            model_dir,
            config=self.model_config,
            #quantization_config=bnb_config, # TODO: only available for GPUs
            device_map='auto'
        )
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(model_dir)
        self.model.eval()

        # Stopping criteria
        stop_list = ['\nHuman:', '\n```\n']
        stop_token_ids = [self.tokenizer(x)['input_ids'] for x in stop_list]
        stop_token_ids = [torch.LongTensor(x).to(device) for x in stop_token_ids]
        self.stopping_criteria = StoppingCriteriaList([StopOnTokens(stop_token_ids)])

        # Ensuring reasonable text generation
        self.generate_text = transformers.pipeline(
            model=self.model, 
            tokenizer=self.tokenizer,
            return_full_text=True,
            task='text-generation',
            stopping_criteria=self.stopping_criteria,
            temperature=0.1,
            max_new_tokens=512,
            repetition_penalty=1.1
        )

llama_7b_chat = Llama7BChatQAInferenceRay()
res = llama_7b_chat.generate_text('Explain me the difference between Data Lakehouse and Data Warehouse.')
print(res[0]['generated_text'])
