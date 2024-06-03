import torch
import torchvision.models as models
from torchvision import transforms
from PIL import Image
from torch_tensorrt import TRTModule
import time

# loads pre-trained resnet50 model
def load_model():
    model = models.resnet50(pretrained=True).eval()
    return model

# preprocesses image for resnet50
def preprocess_image(image_path):
    preprocess = transforms.Compose([
        transforms.Resize(256),                   # Resize to 256x256
        transforms.CenterCrop(224),               # Crop the center 224x224
        transforms.ToTensor(),                    # Convert to Tensor
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],           # Normalize using ImageNet mean
            std=[0.229, 0.224, 0.225]             # Normalize using ImageNet std
        ),
    ])
    image = Image.open(image_path)                # Load image
    image = preprocess(image).unsqueeze(0)        # Preprocess and add batch dimension
    return image

# performs inference 
def benchmark(model, image_path):
    model_trt = TRTModule(model)
    image = preprocess_image(image_path)
    with torch.no_grad():
        output = model_trt(image)
    return output

if __name__ == '__main__':
    model = load_model()
    image_path = 'path_to_your_image.jpg'  # TODO: replace with my image path
    num_iterations = 100

    # Warmup
    output = benchmark(model, image_path)

    # Start benchmarking
    start_time = time.time()
    for _ in range(num_iterations):
        output = benchmark(model, image_path)
    end_time = time.time()

    # TODO: add nvsmi reading functionality

    # Output results
    print(f'Total time for {num_iterations} iterations: {elapsed_time:.4f} seconds')
    print(f'Average time per iteration: {elapsed_time / num_iterations:.4f} seconds')

