import joblib
import s3fs
import sklearn

@serve.deployment(route_prefix='/sentiment', name='sentiment-deployment')
class SentimentDeployment:
    def __init__(self):
        fs = s3fs.S3FileSystem(anon=True)
    with fs.open('ray-serve-blog/unigram_vectorizer.joblib', 'rb') as f:
        self.vectorizer = joblib.load(f)
        f
