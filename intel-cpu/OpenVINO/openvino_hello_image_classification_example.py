# OpenVINO tutorial from https://docs.openvino.ai/2024/notebooks/001-hello-world-with-output.html
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import openvino as ov
import ipywidgets as widgets

import urllib.request
urllib.request.urlretrieve(
    url='https://raw.githubusercontent.com/openvinotoolkit/openvino_notebooks/main/notebooks/utils/notebook_utils.py',
    filename='notebook_utils.py'
)

from notebook_utils import download_file


base_artifacts_dir = Path('./artifacts').expanduser()
model_name = "v3-small_224_1.0_float"
model_xml_name = f'{model_name}.xml'
model_bin_name = f'{model_name}.bin'

model_xml_path = base_artifacts_dir / model_xml_name
base_url = 'https://storage.openvinotoolkit.org/repositories/openvino_notebooks/models/mobelinet-v3-tf/FP32/'

if not model_xml_path.exists():
    download_file(base_url + model_xml_name, model_xml_name, base_artifacts_dir)
    download_file(base_url + model_bin_name, model_bin_name, base_artifacts_dir)
else:
    print(f'{model_name} already downloaded to {base_artifacts_dir}')


core = ov.Core()
device = widgets.Dropdown(
    options=core.available_devices + ["AUTO"],
    value='AUTO',
    description='Device:',
    disabled=False,
)
print(device)


core = ov.Core()
model = core.read_model(model=model_xml_path)
compiled_model = core.compile_model(model=model, device_name=device.value)
output_layer = compiled_model.output(0)


image_filename = download_file(
    "https://storage.openvinotoolkit.org/repositories/openvino_notebooks/data/data/image/coco.jpg",
    directory="data"
)
image = cv2.cvtColor(cv2.imread(filename=str(image_filename)), code=cv2.COLOR_BGR2RGB)
input_image = cv2.resize(src=image, dsize=(224, 224))
input_image = np.expand_dims(input_image, 0)


result_infer = compiled_model([input_image])[output_layer]
result_index = np.argmax(result_infer)
imagenet_filename = download_file(
    "https://storage.openvinotoolkit.org/repositories/openvino_notebooks/data/data/datasets/imagenet/imagenet_2012.txt",
    directory="data"
)
imagenet_classes = imagenet_filename.read_text().splitlines()


imagenet_classes = ['background'] + imagenet_classes
print(imagenet_classes[result_index])
