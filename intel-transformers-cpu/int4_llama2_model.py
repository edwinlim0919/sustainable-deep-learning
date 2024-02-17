#import transformers
from transformers import AutoTokenizer, TextStreamer
from intel_extension_for_transformers.transformers import AutoModelForCausalLM
#from intel_extension_for_transformers.transformers.pipeline import pipeline
#from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline


model_name = 'TheBloke/Llama-2-7B-Chat-GGUF'
model_file = 'llama-2-7b-chat.Q4_0.gguf'
tokenizer_name = 'meta-llama/Llama-2-7b-chat-hf'

prompt = "Once upon a time"
tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)
inputs = tokenizer(prompt, return_tensors="pt").input_ids
streamer = TextStreamer(tokenizer)
model = AutoModelForCausalLM.from_pretrained(model_name, model_file = model_file)
outputs = model.generate(inputs, streamer=streamer, max_new_tokens=300)

#model_config = transformers.AutoConfig.from_pretrained(model_name, model_file=model_file)

#print('model_config: ' + str(model_config))

#prompt = 'Once upon a time'
#tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)
#inputs = tokenizer(prompt, return_tensors="pt").input_ids
#
#model = AutoModelForCausalLM.from_pretrained(
#    model_name,
#    model_file=model_file,
#    trust_remote_code=True,
#    config=model_config
#)


#text_classifier = pipeline(
#    task='text-classification',
#    model='TheBloke/Llama-2-7B-Chat-GGUF',
#    model_file='llama-2-7b-chat.Q4_0.gguf'
#)


#pipe = HuggingFacePipeline(pipeline=pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=128))

#streamer = TextStreamer(tokenizer)
#outputs = model.generate(inputs, streamer=streamer, max_new_tokens=300)


#model = transformers.AutoModelForCausalLM.from_pretrained(
#    model_name,
#    model_file=model_file,
#    trust_remote_code=True,
#    config=model_config
#)


#generate_text = pipeline(
#    model=model,
#    config=model_config,
#    tokenizer=tokenizer,
#    return_full_text=True,
#    task='text-generation',
#    temperature=0.0,
#    max_new_tokens=512,
#    repetition_penalty=1.1
#)

#pipe = HuggingFacePipeline(pipeline=pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=128))
#retriever = VectorStoreRetriever(vectorstore=knowledge_base, search_type='mmr', search_kwargs={'k':1, 'fetch_k':5})
#retrievalQA = RetrievalQA.from_llm(llm=pipe, retriever=retriever)
#
#generate_text = pipeline(
#    model=
#)
