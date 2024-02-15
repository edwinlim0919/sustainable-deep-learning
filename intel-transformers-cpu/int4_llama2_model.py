from transformers import AutoTokenizer, TextStreamer
from intel_extension_for_transformers.transformers import AutoModelForCausalLM

model_name = 'TheBloke/Llama-2-7B-Chat-GGUF'
model_file = 'llama-2-7b-chat.Q4_0.gguf'
tokenizer_name = 'meta-llama/Llama-2-7b-chat-hf'

prompt = 'Once upon a time'
tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)
inputs = tokenizer(prompt, return_tensors="pt").input_ids
streamer = TextStreamer(tokenizer)
model = AutoModelForCausalLM.from_pretrained(model_name, model_file = model_file)
outputs = model.generate(inputs, streamer=streamer, max_new_tokens=300)
