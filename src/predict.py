# ========================================================================
# PREDICTION SCRIPT: src/predict.py
# ========================================================================

# !/usr/bin/env python3
"""
MNIST Deep Learning Classification - Prediction Script

Professional inference pipeline for handwritten digit recognition.

Usage:
    python src/predict.py --image path/to/image.png --model models/best_model.pth --gpu
    python src/predict.py --image path/to/image.png --model models/best_model.pth --cpu
"""

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
import argparse
import numpy as np

from src.models import MNISTClassifier, ImprovedMNISTClassifier
from src.utils import get_device


def load_and_preprocess_image(image_path):
    """
    Load and preprocess a single image for prediction.

    Args:
        image_path (str): Path to the image file

    Returns:
        torch.Tensor: Preprocessed image tensor
    """

    # Define transforms (same as training)
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    # Load and transform image
    image = Image.open(image_path)
    image_tensor = transform(image).unsqueeze(0)  # Add batch dimension

    return image_tensor


def predict_digit(model, image_tensor, device, top_k=1):
    """
    Predict digit from image tensor.

    Args:
        model: Trained PyTorch model
        image_tensor: Preprocessed image tensor
        device: Computing device
        top_k: Number of top predictions to return

    Returns:
        tuple: (predictions, probabilities)
    """

    model.eval()
    image_tensor = image_tensor.to(device)

    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = F.softmax(outputs, dim=1)

        # Get top-k predictions
        top_probs, top_indices = torch.topk(probabilities, top_k, dim=1)

        predictions = top_indices.squeeze().cpu().numpy()
        probs = top_probs.squeeze().cpu().numpy()

        if top_k == 1:
            predictions = [predictions]
            probs = [probs]

    return predictions, probs


def main():
    """Main prediction function."""

    parser = argparse.ArgumentParser(description='MNIST Digit Prediction')
    parser.add_argument('--image', required=True, help='Path to input image')
    parser.add_argument('--model', required=True, help='Path to model checkpoint')
    parser.add_argument('--top_k', default=1, type=int, help='Number of top predictions')
    parser.add_argument('--confidence', action='store_true', help='Show confidence scores')
    # Device selection - mutually exclusive group
    device_group = parser.add_mutually_exclusive_group()
    device_group.add_argument('--gpu', action='store_true', help='Use GPU if available')
    device_group.add_argument('--cpu', action='store_true', help='Force use of CPU')

    args = parser.parse_args()

    # Handle device selection logic
    if args.cpu:
        use_gpu = False
    elif args.gpu:
        use_gpu = True
    else:
        # Default behavior: try GPU if available, fallback to CPU
        use_gpu = True

    # Setup
    device = get_device(use_gpu)

    # Load model
    print("Loading model...")
    checkpoint = torch.load(args.model, map_location=device)

    # Determine model architecture
    if 'architecture' in checkpoint:
        arch = checkpoint['architecture']
    else:
        # Try to infer from model_state_dict keys
        state_dict = checkpoint['model_state_dict']
        if 'bn1.weight' in state_dict:
            arch = 'improved'
        else:
            arch = 'base'

    # Create model
    if arch == 'improved':
        model = ImprovedMNISTClassifier()
    else:
        model = MNISTClassifier()

    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)

    print(f"Model loaded: {arch} architecture")
    if 'accuracy' in checkpoint:
        print(f"Model accuracy: {checkpoint['accuracy']:.2f}%")

    # Load and preprocess image
    print(f"Loading image: {args.image}")
    image_tensor = load_and_preprocess_image(args.image)

    # Make prediction
    predictions, probabilities = predict_digit(model, image_tensor, device, args.top_k)

    # Display results
    print("\nPrediction Results:")
    print("=" * 30)

    for i, (pred, prob) in enumerate(zip(predictions, probabilities)):
        if args.confidence:
            print(f"#{i + 1}: Digit {pred} (confidence: {prob:.3f})")
        else:
            print(f"#{i + 1}: Digit {pred}")

    print(f"\nMost likely digit: {predictions[0]}")


if __name__ == '__main__':
    main()