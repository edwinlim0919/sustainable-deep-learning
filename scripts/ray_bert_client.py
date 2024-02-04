import requests


base_response = requests.post(
    'http://127.0.0.1:8000/bert/baseinference',
    json={
        'inference_type': 'BertBase',
        'inference_text': 'granola bars'
    }
)
print('base_response.text: ' + str(base_response.text))


qa_response = requests.post(
    'http://127.0.0.1:8000/bert/qainference',
    json={
        'inference_type': 'BertQA',
        'inference_question': 'Who was Jim Henson?',
        'inference_text': "Jim Henson was a nice puppet"
    }
)
print('qa_response.text: ' + str(qa_response.text))
