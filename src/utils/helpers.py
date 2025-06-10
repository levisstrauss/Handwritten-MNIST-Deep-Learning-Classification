import torch
import numpy as np
import random
import os


def set_seed(seed=42):
    """
    Set random seeds for reproducibility.

    Args:
        seed (int): Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def save_model(model, optimizer, epoch, val_acc, val_loss, filepath):
    """
    Save model checkpoint with comprehensive information.

    Args:
        model: PyTorch model
        optimizer: Optimizer state
        epoch: Current epoch
        val_acc: Validation accuracy
        val_loss: Validation loss
        filepath: Path to save the checkpoint
    """

    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_acc': val_acc,
        'val_loss': val_loss,
        'model_architecture': model.__class__.__name__,
    }

    torch.save(checkpoint, filepath)
    print(f"Model saved to {filepath}")


def load_model(model, filepath, device):
    """
    Load model checkpoint.

    Args:
        model: PyTorch model instance
        filepath: Path to the checkpoint
        device: Computing device

    Returns:
        dict: Checkpoint information
    """

    checkpoint = torch.load(filepath, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])

    print(f"Model loaded from {filepath}")
    print(f"Validation accuracy: {checkpoint['val_acc']:.2f}%")

    return checkpoint


def get_device(use_gpu=True):
    """
    Get the appropriate computing device.

    Args:
        use_gpu (bool): Whether to use GPU if available

    Returns:
        torch.device: Computing device
    """

    if use_gpu and torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device('cpu')
        print("Using CPU")

    return device
