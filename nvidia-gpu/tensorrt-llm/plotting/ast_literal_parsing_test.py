import ast

test_str = "{'timestamp_readable': '00:21:00', 'timestamp_raw': 1713759660.0, 0: {'temp_celsius': '43C', 'power_usage': '233W / 250W', 'memory_usage': '30624MiB / 32768MiB', 'gpu_utilization': '100%'}, 1: {'temp_celsius': '21C', 'power_usage': '24W / 250W', 'memory_usage': '4MiB / 32768MiB', 'gpu_utilization': '0%'}}"
test_dict = ast.literal_eval(test_str)

for key, value in test_dict.items():
    print(f'{key}: {value}')
