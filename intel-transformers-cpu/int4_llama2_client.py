import requests


llama2_base_response = requests.post(
    'http://127.0.0.1:8000/llama2',
    json={
        'prompt': 'What are the ingredients of olio de aglio? I do not want the entire recipe, only a list of ingredients.'
    }
)
print('llama2_base_response.text: ' + str(llama2_base_response.text))
