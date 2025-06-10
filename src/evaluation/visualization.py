import torch
import matplotlib.pyplot as plt
import numpy as np


def show_sample_images(data_loader, num_images=5):
    """
    Display sample images from the dataset.

    Args:
        data_loader: DataLoader with images
        num_images: Number of images to display
    """

    dataiter = iter(data_loader)
    batch = next(dataiter)
    labels = batch[1][:num_images]
    images = batch[0][:num_images]

    fig, axes = plt.subplots(1, num_images, figsize=(12, 3))
    if num_images == 1:
        axes = [axes]

    for i in range(num_images):
        image = images[i].numpy()
        axes[i].imshow(image.squeeze(), cmap='gray')
        axes[i].set_title(f'Label: {int(labels[i])}', fontweight='bold')
        axes[i].axis('off')

    plt.tight_layout()
    plt.show()


def visualize_predictions(model, test_loader, device, num_images=8):
    """
    Visualize model predictions on test images.

    Args:
        model: Trained PyTorch model
        test_loader: Test data loader
        device: Computing device
        num_images: Number of images to visualize
    """

    model.eval()
    dataiter = iter(test_loader)
    images, labels = next(dataiter)

    # Get predictions
    with torch.no_grad():
        images_gpu = images.to(device)
        outputs = model(images_gpu)
        _, predicted = torch.max(outputs, 1)

        # Get confidence scores
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        confidence_scores = torch.max(probabilities, 1)[0]

    # Plot results
    fig, axes = plt.subplots(2, 4, figsize=(15, 8))
    axes = axes.ravel()

    for i in range(min(num_images, len(images))):
        image = images[i].numpy()
        true_label = labels[i].item()
        pred_label = predicted[i].item()
        confidence = confidence_scores[i].item()

        axes[i].imshow(image.squeeze(), cmap='gray')

        # Color: green if correct, red if incorrect
        color = 'green' if true_label == pred_label else 'red'
        axes[i].set_title(f'True: {true_label}, Pred: {pred_label}\n'
                          f'Confidence: {confidence:.3f}',
                          color=color, fontweight='bold')
        axes[i].axis('off')

    plt.tight_layout()
    plt.show()
