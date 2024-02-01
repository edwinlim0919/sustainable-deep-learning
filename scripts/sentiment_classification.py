from transformers import pipeline
#import joblib
#import s3fs
#import sklearn

@serve.deployment(route_prefix='/sentiment', name='sentiment')
class SentimentDeployment:
    def __init__(self):
        self.classifier = pipeline('sentiment-analysis')

    async def __call__(self, request):
        data = await request.body()
        [result] = self.classifier(str(data))
        return result['label']

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
