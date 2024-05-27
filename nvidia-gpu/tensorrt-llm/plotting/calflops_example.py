from transformers import AutoTokenizer, AutoModelForCausalLM 
from calflops import calculate_flops_hf 


model_name = "meta-llama/Llama-2-7b-hf"
hf_access_token = "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" 
batch_size = 1
sequence_length = 256

flops, macs, params = calculate_flops_hf(
    model_name,
    access_token=hf_access_token,
    input_shape=(batch_size, sequence_length)
)
