from nano_llm import NanoLLM

model = NanoLLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    api="mlc",
    quantization='q4f16_ft'
)

response = model.generate("Once upon a time,", max_new_tokens=128)

for token in response:
    print(token, end='', flush=True)
