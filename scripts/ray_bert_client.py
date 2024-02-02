import requests


cl4_response = requests.post('http://127.0.0.1:8000/bert/cl4inference', params={'text': 'granola bars'})
sentence_embedding = cl4_response.json()
print(sentence_embedding)
