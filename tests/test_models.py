# ========================================================================
# tests/test_models.py
# ========================================================================

import unittest
import torch
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from models import MNISTClassifier, ImprovedMNISTClassifier


class TestModels(unittest.TestCase):
    """Test cases for model architectures."""

    def setUp(self):
        """Set up test fixtures."""
        self.batch_size = 32
        self.input_shape = (self.batch_size, 1, 28, 28)
        self.dummy_input = torch.randn(self.input_shape)

    def test_base_model_forward(self):
        """Test base model forward pass."""
        model = MNISTClassifier()
        output = model(self.dummy_input)

        # Check output shape
        self.assertEqual(output.shape, (self.batch_size, 10))

        # Check output is finite
        self.assertTrue(torch.isfinite(output).all())

    def test_improved_model_forward(self):
        """Test improved model forward pass."""
        model = ImprovedMNISTClassifier()
        output = model(self.dummy_input)

        # Check output shape
        self.assertEqual(output.shape, (self.batch_size, 10))

        # Check output is finite
        self.assertTrue(torch.isfinite(output).all())

    def test_model_training_mode(self):
        """Test model behavior in training vs eval mode."""
        model = ImprovedMNISTClassifier()

        # Training mode
        model.train()
        output_train = model(self.dummy_input)

        # Eval mode
        model.eval()
        output_eval = model(self.dummy_input)

        # Outputs should be different due to dropout/batch norm
        self.assertFalse(torch.allclose(output_train, output_eval, atol=1e-6))

    def test_model_parameters(self):
        """Test model parameter counts."""
        base_model = MNISTClassifier()
        improved_model = ImprovedMNISTClassifier()

        base_params = sum(p.numel() for p in base_model.parameters())
        improved_params = sum(p.numel() for p in improved_model.parameters())

        # Improved model should have more parameters
        self.assertGreater(improved_params, base_params)

        # Reasonable parameter counts
        self.assertGreater(base_params, 100000)  # At least 100K params
        self.assertLess(improved_params, 10000000)  # Less than 10M params