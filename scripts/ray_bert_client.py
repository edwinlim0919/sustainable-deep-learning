import requests


bert_base_response = requests.post(
    'http://127.0.0.1:8000/bert',
    json={
        'inference_type': 'BertBase',
        'inference_text': 'granola bars'
    }
)
print('BERT base_response.text: ' + str(bert_base_response.text))


bert_qa_response = requests.post(
    'http://127.0.0.1:8000/bert',
    json={
        'inference_type': 'BertQA',
        'inference_question': 'Who was Jim Henson?',
        'inference_text': "Jim Henson was a nice puppet"
    }
)
print('BERT qa_response.text: ' + str(bert_qa_response.text))
