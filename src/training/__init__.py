"""
Training Utilities and Optimization Strategies

Contains training loops, optimizers, and learning rate scheduling.
"""

from .trainer import Trainer
from .optimizer import get_optimizer_and_scheduler

__all__ = ['Trainer', 'get_optimizer_and_scheduler']
