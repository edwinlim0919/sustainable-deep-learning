import requests


cl4_response = requests.post('http://127.0.0.1:8000/bert/cl4inference', params={'text': 'granola bars'})
cl4_sentence_embedding = cl4_response.json()
print('cl4_sentence_embedding: ' + str(cl4_sentence_embedding))


qa_response = requests.post('http://127.0.0.1:8000/bert/qainference', params={'question': 'Who was Jim Henson?', 'text': "Jim Henson was a nice puppet"})
qa_answer = qa_response.json()
print('qa_answer: ' + str(qa_answer))
