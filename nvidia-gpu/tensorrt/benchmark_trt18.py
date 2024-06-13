import torch
import torchvision.models as models
import torch_tensorrt
from torchvision import transforms
from PIL import Image
import os
import random
import time
import numpy as np
import argparse
from torchvision.models import ResNet18_Weights
from pathlib import Path


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
    image = Image.open(image_path).convert('RGB') # Load image
    image = preprocess(image)                     # Apply preprocessing
    return image

# Get list of image files from a directory
def get_image_files(directory):
    image_files = [f for f in os.listdir(directory) if f.endswith('.JPEG')]
    return image_files

# performs inference
def benchmark(model, image_directory, output_file, max_batch_size=1, dtype='fp32', nwarmup=50, nruns=100):
    image_files = get_image_files(image_directory)
    if not image_files:
        raise ValueError(f"No image files found in {image_directory}")

    with open(output_file, 'w') as f_out:
        f_out.write(f"num_iterations: {nruns}\n")
        
        f_out.write(f"batch_size: {max_batch_size}\n\n")

        print("Warm up ...")
        with torch.no_grad():
            for _ in range(nwarmup):
                batch_files = random.sample(image_files, max_batch_size)
                images = [preprocess_image(os.path.join(image_directory, img_file)).to('cuda') for img_file in batch_files]
                batch = torch.stack(images)
                features = model(batch)
                torch.cuda.synchronize()

        print("Start timing ...")
        timings = []
        with torch.no_grad():
            for i in range(1, nruns + 1):
                start_time = time.time()
                batch_files = random.sample(image_files, max_batch_size)
                images = [preprocess_image(os.path.join(image_directory, img_file)).to('cuda') for img_file in batch_files]
                batch = torch.stack(images)
                features = model(batch)
                torch.cuda.synchronize()
                end_time = time.time()
                inference_time = (end_time - start_time) * 1000  # Convert to milliseconds
                timings.append(inference_time)
                f_out.write(f"iteration: {i}/{nruns}\n")
                f_out.write(f"batch_files: {batch_files}\n")
                f_out.write(f"start_time: {start_time:.2f}\n")  # Start time in seconds
                f_out.write(f"end_time: {end_time:.2f}\n\n")    # End time in seconds
                if i % 10 == 0:
                    print(f'Iteration {i}/{nruns}, ave batch time {np.mean(timings):.2f} ms')
                    
    print("Input shape:", batch.size())
    print("Type of features:", type(features))
    if isinstance(features, tuple):
        for i, feature in enumerate(features):
            print(f"Output feature {i} size:", feature.size())
    else:
        print("Output features size:", features.size())
    print(f'Average batch time: {np.mean(timings):.2f} ms')
    return timings

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Benchmark ResNet18 with TensorRT')
    parser.add_argument(
        '--batch-size', 
        type=int, 
        default=1, 
        help='Max batch size for inference'
    )
    parser.add_argument(
        '--image-directory', 
        type=str, 
        required=True, 
        help='Directory containing images'
    )
    parser.add_argument(
        '--output-file', 
        type=str, 
        default='output.csv', 
        help='File to write the output'
    )
    parser.add_argument(
        '--num-iterations', 
        type=int, 
        default=100, 
        help='Number of iterations to run'
    )
    parser.add_argument(
        '--container_output_dir',
        type=str,
        required=True,
        help='directory in docker container for saving output files'
    )
    parser.add_argument(
        '--container_stop_file',
        type=str,
        required=True,
        help='filepath in docker container for coordinating nvsmi loop stop'
    )
    
    args = parser.parse_args()

    model = load_model()
    
    trt_model_fp32 = torch_tensorrt.compile(model, 
                                            inputs=[torch_tensorrt.Input(
                                                shape=(args.batch_size, 3, 224, 224),  # Maximum dynamic shape
                                                dtype=torch.float32
                                            )],
                                            enabled_precisions={torch.float32},  # Run with FP32
                                            workspace_size=1 << 22)

    timings = benchmark(trt_model_fp32, args.image_directory, args.output_file, max_batch_size=args.batch_size, nruns=args.num_iterations)

    # Output results
    if timings:
        total_time = sum(timings) / 1000  # Convert to seconds
        avg_time_per_iteration = np.mean(timings)
        print(f'Total time for {args.num_iterations} iterations: {total_time:.4f} seconds')
        print(f'Average time per iteration: {avg_time_per_iteration:.4f} ms')
    else:
        print("No timings recorded during benchmarking.")
        
    container_output_dir = Path(args.container_output_dir)
    container_output_dir.mkdir(exist_ok=True, parents=True)
    with (container_output_dir / args.container_stop_file).open('w') as f:
        f.write('COMPLETED\n')

