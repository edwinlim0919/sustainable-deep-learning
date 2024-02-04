import requests


cl4_response = requests.post(
    'http://127.0.0.1:8000/bert/cl4inference',
    json={
        'inference_type': 'BertCL4',
        'inference_text': 'granola bars'
    }
)
print('cl4_response.text: ' + str(cl4_response.text))


qa_response = requests.post(
    'http://127.0.0.1:8000/bert/qainference',
    json={
        'inference_type': 'BertQA',
        'inference_question': 'Who was Jim Henson?',
        'inference_text': "Jim Henson was a nice puppet"
    }
)
print('qa_response.text: ' + str(qa_response.text))
