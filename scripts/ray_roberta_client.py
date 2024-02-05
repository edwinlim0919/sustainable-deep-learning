import requests


roberta_base_response = requests.post(
    'http://127.0.0.1:8000/roberta',
    json={
        'inference_type': 'RobertaBase',
        'inference_text': 'granola bars'
    }
)
print('ROBERTA base_response.text: ' + str(roberta_base_response.text))


#roberta_qa_response = requests.post(
#    'http://127.0.0.1:8000/roberta',
#    json={
#        'inference_type': 'RobertaQA',
#        'inference_question': 'Who was Jim Henson?',
#        'inference_text': "Jim Henson was a nice puppet"
#    }
#)
#print('ROBERTA qa_response.text: ' + str(roberta_qa_response.text))
