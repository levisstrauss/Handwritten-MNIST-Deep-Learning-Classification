"""
Neural Network Model Architectures

Contains base and improved model implementations for MNIST classification.
"""

from .base_model import MNISTClassifier
from .improved_model import ImprovedMNISTClassifier

__all__ = ['MNISTClassifier', 'ImprovedMNISTClassifier']