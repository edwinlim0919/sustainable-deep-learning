import torch
import transformers
from torch import cuda, bfloat16
from transformers import LlamaForCausalLM, LlamaTokenizer
from transformers import StoppingCriteria, StoppingCriteriaList
from langchain.chains import ConversationalRetrievalChain
from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.language_models.llms import LLM
from typing import Any, List, Optional, Mapping

from embeddings_vector_store import EmbeddingsVectorStore


#class Llama7BChatEndpointRay:


# TODO: Can potentially quantify sustainability benefits of quantization
# Enabling quanitzation for large model with less memory requirements
bnb_config = transformers.BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type='nf4',
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=bfloat16
)


# TODO: This is janky but not sure what else to do
global_generate_text = None


# Custom stopping criteria object
class StopOnTokens(StoppingCriteria):
    def __init__(self, stop_token_ids):
        self.stop_token_ids = stop_token_ids

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        for stop_ids in self.stop_token_ids:
            if torch.eq(input_ids[0][-len(stop_ids):], stop_ids).all():
                return True
        return False


class CustomLLM(LLM):
    def __init__(self, model_name): # TODO model_name does not do anything
        model_dir = '../model_weights/llama-2-7b-chat-hf'
        device = f'cuda:{cuda.current_device()}' if cuda.is_available() else 'cpu'
        model_config = transformers.AutoConfig.from_pretrained(model_dir)
        if cuda.is_available():
            model = transformers.AutoModelForCausalLM.from_pretrained(
                model_dir,
                config=model_config,
                quantization_config=bnb_config,
                device_map='auto'
            )
        else:
            model = transformers.AutoModelForCausalLM.from_pretrained(
                model_dir,
                config=model_config,
                #quantization_config=bnb_config, # TODO: only available for GPUs
                device_map='auto'
            )
        tokenizer = transformers.AutoTokenizer.from_pretrained(model_dir)
        model.eval()

        # Stopping criteria
        stop_list = ['\nHuman:', '\n```\n']
        stop_token_ids = [tokenizer(x)['input_ids'] for x in stop_list]
        stop_token_ids = [torch.LongTensor(x).to(device) for x in stop_token_ids]
        stopping_criteria = StoppingCriteriaList([StopOnTokens(stop_token_ids)])

        # Ensuring reasonable text generation
        global_generate_text = transformers.pipeline(
            model=model, 
            tokenizer=tokenizer,
            return_full_text=True,
            task='text-generation',
            stopping_criteria=stopping_criteria,
            temperature=0.1,
            max_new_tokens=512,
            repetition_penalty=1.1
        )

    @property
    def _llm_type(self) -> str:
        return 'custom'

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        kwargs = {
            'temperature' : 0.1,
            'max_new_tokens' : 512,
            'repetition_penalty' : 1.1,
            **kwargs
        }
        return 'fuck'
        #return global_generate_text(input_text, **kwargs)

    # TODO: This is kind of bullshit
    #@property
    #def _identifying_params(self) -> Mapping[str, Any]:
    #    return {'n': 10}


class Llama7BChatPipeline:
    def __init__(self):
        self.llm = CustomLLM('llama-2-7b-chat-hf')
        self.embeddings_vector_store = EmbeddingsVectorStore()
        self.chain = ConversationalRetrievalChain.from_llm(
            self.llm,
            self.embeddings_vector_store.vectorstore.as_retriever(),
            return_source_documents=False
        )
        self.chat_history = []

    def reset_chat_history(self):
        self.chat_history.clear()

    def query(self, query_text):
        return self.llm._call('fuck')
        #result = self.chain({'question' : query_text, 'chat_history' : self.chat_history})
        #self.chat_history.append((query_text, result['answer']))
        #return result['answer']


llama_7b_chat_pipeline = Llama7BChatPipeline()
print(llama_7b_chat_pipeline.query('My name is Edwin.'))
print(llama_7b_chat_pipeline.query('What is my name?'))
