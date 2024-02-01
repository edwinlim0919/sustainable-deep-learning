import ray
from ray import serve
from fastapi import FastAPI

from transformers import pipeline


app = FastAPI()


@serve.deployment(num_replicas=2, ray_actor_options={'num_cpus': 0.2, 'num_gpus': 0})
@serve.ingress(app)
class Translator:
    def __init__(self):
        self.model = pipeline('translation_en_to_fr', model='t5-small')

    @app.post('/')
    def translate(self, text: str) -> str:
        model_output = self.model(text)
        translation = model_output[0]['translation_text']
        return translation


translator_app = Translator.bind()
