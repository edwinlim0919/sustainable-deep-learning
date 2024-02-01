import ray
from ray import serve

# This connects to the local running Ray cluster
ray.init(address='auto', namespace='serve-example', ignore_reinit_error=True)
