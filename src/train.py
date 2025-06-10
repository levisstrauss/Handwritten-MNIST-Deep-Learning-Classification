# ========================================================================
# MAIN TRAINING SCRIPT: src/train.py
# ========================================================================

# !/usr/bin/env python3
"""
MNIST Deep Learning Classification - Training Script

Industry-grade training pipeline for handwritten digit recognition.

Usage:
    python src/train.py --arch improved --epochs 15 --gpu
"""

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn

from src.models import MNISTClassifier, ImprovedMNISTClassifier
from src.data import get_dataloaders
from src.training import Trainer, get_optimizer_and_scheduler
from src.evaluation import evaluate_model, plot_confusion_matrix
from src.utils import Config, setup_logger, set_seed, get_device


def main():
    """Main training function."""

    # Parse arguments
    config = Config()
    args = config.parse_args()

    # Setup
    set_seed(args.seed)
    device = get_device(args.gpu)
    logger = setup_logger()

    logger.info("Starting MNIST Deep Learning Classification Training")
    logger.info(f"Configuration: {vars(args)}")

    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)

    # Load data
    logger.info("Loading MNIST dataset...")
    train_loader, test_loader, viz_loader = get_dataloaders(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        data_dir=args.data_dir
    )

    logger.info(f"Training samples: {len(train_loader.dataset)}")
    logger.info(f"Test samples: {len(test_loader.dataset)}")

    # Create model
    logger.info(f"Creating {args.arch} model...")
    if args.arch == 'base':
        model = MNISTClassifier()
    elif args.arch == 'improved':
        model = ImprovedMNISTClassifier()
    else:
        raise ValueError(f"Unknown architecture: {args.arch}")

    model = model.to(device)

    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer, scheduler = get_optimizer_and_scheduler(
        model,
        optimizer_type=args.optimizer,
        lr=args.learning_rate,
        scheduler_type=args.scheduler if args.scheduler != 'none' else None,
        epochs=args.epochs,
        steps_per_epoch=len(train_loader)
    )

    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    logger.info(f"Optimizer: {args.optimizer}")
    logger.info(f"Scheduler: {args.scheduler}")

    # Create trainer
    trainer = Trainer(model, device, criterion, optimizer, scheduler)

    # Train model
    logger.info("Starting training...")
    train_losses, train_accs, val_losses, val_accs = trainer.train(
        train_loader, test_loader, args.epochs, args.save_dir
    )

    # Final evaluation
    logger.info("Evaluating final model...")
    results = evaluate_model(model, test_loader, device, criterion)

    logger.info(f"Final Test Accuracy: {results['overall_accuracy']:.2f}%")
    logger.info("Per-class accuracies:")
    for digit, acc in results['per_class_accuracy'].items():
        logger.info(f"  Digit {digit}: {acc:.2f}%")

    # Plot results
    trainer.plot_training_history()
    plot_confusion_matrix(results['targets'], results['predictions'])

    # Save final model
    final_model_path = os.path.join(args.save_dir, f'{args.arch}_final_model.pth')
    torch.save({
        'model_state_dict': model.state_dict(),
        'architecture': args.arch,
        'accuracy': results['overall_accuracy'],
        'config': vars(args)
    }, final_model_path)

    logger.info(f"Final model saved to {final_model_path}")
    logger.info("Training completed successfully!")


if __name__ == '__main__':
    main()
