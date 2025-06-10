#!/usr/bin/env python3
"""
Quick script to extract a few MNIST samples for testing.
"""

import os
from PIL import Image
import torchvision.datasets as datasets
import torchvision.transforms as transforms

# Create test images directory
os.makedirs('./test_images', exist_ok=True)

# Load MNIST test dataset
dataset = datasets.MNIST(
    root='./data',
    train=False,
    download=True,
    transform=transforms.ToTensor()
)

print("Extracting 5 sample images...")

# Extract first 5 images
for i in range(5):
    image_tensor, label = dataset[i]

    # Convert to PIL Image
    image_array = (image_tensor.squeeze().numpy() * 255).astype('uint8')
    image = Image.fromarray(image_array, mode='L')

    # Save
    filename = f"./test_images/digit_{label}_sample_{i}.png"
    image.save(filename)
    print(f"Saved: {filename} (actual digit: {label})")