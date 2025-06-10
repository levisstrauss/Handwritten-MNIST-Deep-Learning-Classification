# ========================================================================
# tests/test_data.py
# ========================================================================

import unittest
import torch
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from data import get_dataloaders, get_transforms


class TestDataLoading(unittest.TestCase):
    """Test cases for data loading and preprocessing."""

    def test_transforms(self):
        """Test data transforms."""
        train_transform, test_transform, viz_transform = get_transforms()

        # Create dummy PIL image
        from PIL import Image
        import numpy as np

        dummy_image = Image.fromarray(np.random.randint(0, 255, (28, 28), dtype=np.uint8))

        # Apply transforms
        train_tensor = train_transform(dummy_image)
        test_tensor = test_transform(dummy_image)
        viz_tensor = viz_transform(dummy_image)

        # Check shapes
        self.assertEqual(train_tensor.shape, (1, 28, 28))
        self.assertEqual(test_tensor.shape, (1, 28, 28))
        self.assertEqual(viz_tensor.shape, (1, 28, 28))

        # Check value ranges
        self.assertTrue(torch.all(viz_tensor >= 0) and torch.all(viz_tensor <= 1))

    def test_dataloaders(self):
        """Test dataloader creation."""
        # Use small batch size for testing
        train_loader, test_loader, viz_loader = get_dataloaders(
            batch_size=4, num_workers=0
        )

        # Test train loader
        train_batch = next(iter(train_loader))
        self.assertEqual(len(train_batch), 2)  # images and labels
        self.assertEqual(train_batch[0].shape[0], 4)  # batch size
        self.assertEqual(train_batch[0].shape[1:], (1, 28, 28))  # image shape

        # Test test loader
        test_batch = next(iter(test_loader))
        self.assertEqual(len(test_batch), 2)
        self.assertEqual(test_batch[0].shape[0], 4)

        # Check label ranges
        self.assertTrue(torch.all(train_batch[1] >= 0))
        self.assertTrue(torch.all(train_batch[1] <= 9))
