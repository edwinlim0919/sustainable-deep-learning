import os
import random
import torch
import torchvision.models as models
import torch_tensorrt
from torchvision import transforms
from PIL import Image
import time
import numpy as np
from torchvision.models import ResNet18_Weights

# loads pre-trained resnet18 model
def load_model():
    model = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1).eval().to('cuda')
    return model

# preprocesses image for resnet18
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
    image = preprocess(image)                     # Apply preprocessing
    image = image.unsqueeze(0).to('cuda')         # Add batch dimension and move to GPU
    return image

# Get list of image files from a directory
def get_image_files(directory):
    image_files = [f for f in os.listdir(directory) if f.endswith(('.jpg', '.jpeg', '.png'))]
    return image_files

# performs inference
def benchmark(model, image_directory, output_file, dtype='fp32', nwarmup=50, nruns=100):
    image_files = get_image_files(image_directory)
    if not image_files:
        raise ValueError(f"No image files found in {image_directory}")

    with open(output_file, 'w') as f_out:
        f_out.write("Image,Inference Time (ms)\n")

        print("Warm up ...")
        with torch.no_grad():
            for _ in range(nwarmup):
                # Randomly select an image file for warm-up
                image_file = random.choice(image_files)
                image_path = os.path.join(image_directory, image_file)
                image = preprocess_image(image_path)
                features = model(image)
                torch.cuda.synchronize()

        print("Start timing ...")
        timings = []
        with torch.no_grad():
            for i in range(1, nruns + 1):
                start_time = time.time()
                # Randomly select an image file for each iteration
                image_file = random.choice(image_files)
                image_path = os.path.join(image_directory, image_file)
                image = preprocess_image(image_path)
                features = model(image)
                torch.cuda.synchronize()
                end_time = time.time()
                inference_time = (end_time - start_time) * 1000  # in milliseconds
                timings.append(inference_time)
                f_out.write(f"{image_file},{inference_time:.2f}\n")
                if i % 10 == 0:
                    print(f'Iteration {i}/{nruns}, ave batch time {np.mean(timings):.2f} ms')
                    
    print("Input shape:", image.size())
    # Check and print the type and size of features
    print("Type of features:", type(features))
    if isinstance(features, tuple):
        for i, feature in enumerate(features):
            print(f"Output feature {i} size:", feature.size())
    else:
        print("Output features size:", features.size())
    print(f'Average batch time: {np.mean(timings) * 1000:.2f} ms')
    return timings

if __name__ == '__main__':
    model = load_model()
    trt_model_fp32 = torch_tensorrt.compile(model, 
                                            inputs=[torch_tensorrt.Input((1, 3, 224, 224), dtype=torch.float32)],  # Change batch size to 1
                                            enabled_precisions={torch.float32},  # Run with FP32
                                            workspace_size=1 << 22)

    image_directory = '/workspace/tensorrt/images'  # TODO: replace with your image directory
    output_file = 'output.csv'
    num_iterations = 100

    timings = benchmark(trt_model_fp32, image_directory, output_file, nruns=num_iterations)

    # Output results
    if timings:
        total_time = sum(timings)
        avg_time_per_iteration = np.mean(timings)
        print(f'Total time for {num_iterations} iterations: {total_time:.4f} seconds')
        print(f'Average time per iteration: {avg_time_per_iteration:.4f} seconds')
    else:
        print("No timings recorded during benchmarking.")

