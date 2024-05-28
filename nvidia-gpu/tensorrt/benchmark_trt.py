import argparse
import asyncio
import random
import time
from pathlib import Path

import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image

import benchmark_utils

import tensorrt_llm.logger as logger
from tensorrt_llm.runtime import PYTHON_BINDINGS, ModelRunnerCpp

def eval_resnet(image_path: str, model: torch.nn.Module):
    image = Image.open(image_path)
    # Preprocess the image using torchvision.transforms
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    input_tensor = preprocess(image)
    input_batch = input_tensor.unsqueeze(0)  # Add a batch dimension

    # Run the model
    with torch.no_grad():
        output = model(input_batch)

    return output


async def main(args):
    runtime_rank = tensorrt_llm.mpi_rank()
    logger.set_level(args.log_level)

    # Initialize ResNet model
    resnet_model = models.resnet50(pretrained=True)
    resnet_model.eval()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    container_output_dir = Path(args.container_output_dir)
    container_output_dir.mkdir(exist_ok=True, parents=True)

    for iteration in range(args.num_iterations):
        logger.info(f'Iteration: {iteration + 1}/{args.num_iterations}')
        # Replace this with actual image paths or logic to load images
        image_path = 'path/to/your/image.jpg'
        output = eval_resnet(image_path, resnet_model)

        # Save or log your output (example: printing the output)
        logger.info(output)

    await asyncio.sleep(30)
    with (container_output_dir / args.container_stop_file).open('w') as f:
        f.write('COMPLETED\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--output_dir',
        type=str,
        required=True,
        help='directory for saving output files'
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
    parser.add_argument(
        '--num_iterations',
        type=int,
        required=True,
        help='number of batch iterations to run during profiling'
    )
    parser.add_argument(
        '--log_level',
        type=str,
        default='info',
        help='logging level (default: info)'
    )
    args = parser.parse_args()

    asyncio.run(main(args))
