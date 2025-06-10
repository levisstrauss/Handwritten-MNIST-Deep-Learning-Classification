import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from tqdm import tqdm


def evaluate_model(model, test_loader, device, criterion=None):
    """
    Comprehensive model evaluation.

    Args:
        model: PyTorch model
        test_loader: Test data loader
        device: Computing device
        criterion: Loss function (optional)

    Returns:
        dict: Evaluation metrics
    """

    model.eval()
    test_loss = 0.0
    correct_predictions = 0
    total_samples = 0
    all_predictions = []
    all_targets = []
    class_correct = [0] * 10
    class_total = [0] * 10

    with torch.no_grad():
        for data, targets in tqdm(test_loader, desc='Evaluating'):
            data, targets = data.to(device), targets.to(device)
            outputs = model(data)

            if criterion:
                loss = criterion(outputs, targets)
                test_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            total_samples += targets.size(0)
            correct_predictions += (predicted == targets).sum().item()

            # Per-class accuracy
            c = (predicted == targets).squeeze()
            for i in range(targets.size(0)):
                label = targets[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1

            all_predictions.extend(predicted.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())

    # Calculate metrics
    overall_accuracy = 100. * correct_predictions / total_samples
    per_class_accuracy = {}
    for i in range(10):
        if class_total[i] > 0:
            per_class_accuracy[i] = 100 * class_correct[i] / class_total[i]

    results = {
        'overall_accuracy': overall_accuracy,
        'per_class_accuracy': per_class_accuracy,
        'predictions': all_predictions,
        'targets': all_targets,
        'correct_predictions': correct_predictions,
        'total_samples': total_samples
    }

    if criterion:
        results['test_loss'] = test_loss / len(test_loader)

    return results


def plot_confusion_matrix(targets, predictions, save_path=None):
    """
    Plot confusion matrix with professional styling.

    Args:
        targets: True labels
        predictions: Predicted labels
        save_path: Path to save the plot (optional)
    """

    cm = confusion_matrix(targets, predictions)

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=range(10), yticklabels=range(10),
                cbar_kws={'label': 'Count'})

    plt.title('Confusion Matrix - MNIST Digit Classification',
              fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Predicted Label', fontsize=14)
    plt.ylabel('True Label', fontsize=14)

    # Add accuracy text
    accuracy = np.trace(cm) / np.sum(cm) * 100
    plt.text(0.5, -0.1, f'Overall Accuracy: {accuracy:.2f}%',
             transform=plt.gca().transAxes, ha='center', fontsize=12)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.show()


def print_classification_report(targets, predictions):
    """Print detailed classification report."""
    print("\nDetailed Classification Report:")
    print("=" * 50)
    print(classification_report(targets, predictions,
                                target_names=[f'Digit {i}' for i in range(10)]))