import time
import datetime
import requests


start_time = time.time()

llama2_base_response = requests.post(
    'http://127.0.0.1:8000/llama2',
    json={
        'prompt': 'What are the ingredients of olio de aglio? I do not want the entire recipe, only a list of ingredients.'
    }
)

end_time = time.time()
elapsed_time_seconds = end_time - start_time
elapsed_time_readable = str(datetime.timedelta(seconds=elapsed_time_seconds))

print(f'E2E Elapsed Time: {elapsed_time_readable} (hours:min:seconds)')
print('llama2_base_response.text: ' + str(llama2_base_response.text))
