import ray
from ray import serve
from fastapi import FastAPI

from transformers import pipeline
#import joblib
#import s3fs
#import sklearn


app = FastAPI()


@serve.deployment(num_replicas=2, ray_actor_options={'num_cpus': 0.2, 'num_gpus': 0})
@serve.ingress(app)
class Translator:
    def __init__(self):
        self.model = pipeline('translation_en_to_fr', model='t5-small')

    @app.post("/")
    def translate(self, text: str) -> str:
        model_output = self.model(text)
        translation = model_output[0]['translation_text']
        return translation


#translator = Translator()
#translation = translator.translate('Hello world!')
#print(translation)


translator_app = Translator.bind()


#@serve.deployment(route_prefix='/sentiment', name='sentiment')
#class SentimentDeployment:
#    def __init__(self):
#        self.classifier = pipeline('sentiment-analysis')
#
#    async def __call__(self, request):
#        data = await request.body()
#        [result] = self.classifier(str(data))
#        return result['label']

#class SentimentDeployment:
#    def __init__(self):
#        fs = s3fs.S3FileSystem(anon=True)
#        with fs.open('ray-serve-blog/unigram_vectorizer.joblib', 'rb') as f:
#            self.vectorizer = joblib.load(f)
#        with fs.open('ray-serve-blog/unigram_tf_idf_transformer.joblib', 'rb') as f:
#            self.preprocessor = joblib.load(f)
#        with fs.open('ray-serve-blog/unigram_tf_idf_classifier.joblib', 'rb') as f:
#            self.classifier = joblib.load(f)
#
#    async def __call__(self, request):
#        data = await request.body()
#        vectorized = self.vectorizer.transform([str(data)])
#        transformed = self.preprocessor.transform(vectorized)
#        [result] = self.classifier.predict(transformed)
#        if result == 1:
#            return 'POSITIVE'
#        else:
#            return 'NEGATIVE'
