import torch
import torchvision.models as models
import torch_tensorrt
from torchvision import transforms
from PIL import Image
import time
import numpy as np
from torchvision.models import ResNet50_Weights

# loads pre-trained resnet50 model
def load_model():
    model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1).eval().to('cuda')
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
def benchmark(model, image_path, dtype='fp32', nwarmup=50, nruns=100):
    image = preprocess_image(image_path)
    input_data = image.to('cuda')

    if dtype == 'fp16':
        input_data = input_data.half()

    print("Warm up ...")
    with torch.no_grad():
        for _ in range(nwarmup):
            features = model(input_data)
    torch.cuda.synchronize()

    print("Start timing ...")
    timings = []
    with torch.no_grad():
        for i in range(1, nruns + 1):
            start_time = time.time()
            features = model(input_data)
            torch.cuda.synchronize()
            end_time = time.time()
            timings.append(end_time - start_time)
            if i % 10 == 0:
                print(f'Iteration {i}/{nruns}, ave batch time {np.mean(timings) * 1000:.2f} ms')

    print("Input shape:", input_data.size())
    print("Output features size:", features.size())
    print(f'Average batch time: {np.mean(timings) * 1000:.2f} ms')

if __name__ == '__main__':
    model = load_model()
    trt_model_fp32 = torch_tensorrt.compile(model, 
                                            inputs=[torch_tensorrt.Input((128, 3, 224, 224), dtype=torch.float32)],
                                            enabled_precisions={torch.float32},  # Run with FP32
                                            workspace_size=1 << 22)

    image_path = '/dev/shm/sustainable-deep-learning/nvidia-gpu/tensorrt/n02352591_3694.JPEG'  # TODO: replace with your image path
    num_iterations = 100

    benchmark(trt_model_fp32, image_path, nruns=num_iterations)

    # TODO: add nvsmi reading functionality

    # Output results
    print(f'Total time for {num_iterations} iterations: {sum(timings):.4f} seconds')
    print(f'Average time per iteration: {np.mean(timings):.4f} seconds')
