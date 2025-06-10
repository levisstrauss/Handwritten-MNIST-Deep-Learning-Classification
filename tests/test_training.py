# ========================================================================
# tests/test_training.py
# ========================================================================

import unittest
import torch
import torch.nn as nn
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from models import MNISTClassifier
from training import get_optimizer_and_scheduler
from utils import get_device


class TestTraining(unittest.TestCase):
    """Test cases for training components."""

    def setUp(self):
        """Set up test fixtures."""
        self.model = MNISTClassifier()
        self.device = get_device(use_gpu=False)  # Use CPU for testing

    def test_optimizer_creation(self):
        """Test optimizer creation."""
        optimizer, scheduler = get_optimizer_and_scheduler(
            self.model, optimizer_type='adam', lr=0.001
        )

        self.assertIsNotNone(optimizer)
        self.assertEqual(len(optimizer.param_groups), 1)
        self.assertEqual(optimizer.param_groups[0]['lr'], 0.001)

    def test_scheduler_creation(self):
        """Test scheduler creation."""
        optimizer, scheduler = get_optimizer_and_scheduler(
            self.model, optimizer_type='adam', scheduler_type='step'
        )

        self.assertIsNotNone(scheduler)

    def test_loss_computation(self):
        """Test loss computation."""
        criterion = nn.CrossEntropyLoss()

        # Create dummy data
        batch_size = 4
        dummy_input = torch.randn(batch_size, 1, 28, 28)
        dummy_targets = torch.randint(0, 10, (batch_size,))

        # Forward pass
        outputs = self.model(dummy_input)
        loss = criterion(outputs, dummy_targets)

        # Check loss properties
        self.assertTrue(torch.isfinite(loss))
        self.assertGreater(loss.item(), 0)
