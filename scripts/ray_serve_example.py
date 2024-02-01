import ray
from ray import serve

import sentiment_classification

# This connects to the locally running Ray cluster
ray.init(address='auto', namespace='serve-example', ignore_reinit_error=True)

# This starts the Ray Serve processes within the Ray cluster
serve.start(detached=True)

# This deploys the model
SentimentDeployment.deploy()
