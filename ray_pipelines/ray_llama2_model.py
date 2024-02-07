import torch
import transformers
from transformers import LlamaForCausalLM, LlamaTokenizer
from transformers import StoppingCriteria, StoppingCriteriaList
#from torch import bfloat16


#class Llama7BChatEndpointRay:


# TODO: Should make this optional, can potentially quantify sustainability benefits of quantization
# Enabling quanitzation for large model with less memory requirements
bnb_config = transformers.BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type='nf4',
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=bfloat16
)


class Llama7BChatQAInferenceRay:
    def __init__(self):
        model_dir = '../model_weights/llama-2-7b-chat-hf'
        self.model = LlamaForCausalLM.from_pretrained(model_dir)
        self.tokenizer = LlamaTokenizer.from_pretrained(model_dir)
        self.model.eval()

# Stopping criteria
stop_list = ['\nHuman:', '\n```\n']
stop_token_ids = [tokenizer(x)['input_ids'] for x in stop_list]
stop_token_ids = [torch.LongTensor(x).to(device) for x in stop_token_ids]
