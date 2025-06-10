import torch
import torch.nn as nn
import torch.nn.functional as F


class MNISTClassifier(nn.Module):
    """
    Base MNIST Classifier with standard architecture.

    Features:
    - 3 hidden layers with progressive dimension reduction
    - ReLU activations with dropout regularization
    - Xavier weight initialization

    Architecture: 784 → 512 → 256 → 128 → 10
    """

    def __init__(self, input_size=784, hidden_sizes=[512, 256, 128],
                 num_classes=10, dropout_rate=0.3):
        super(MNISTClassifier, self).__init__()

        self.flatten = nn.Flatten()

        # Network layers
        self.fc1 = nn.Linear(input_size, hidden_sizes[0])
        self.dropout1 = nn.Dropout(dropout_rate)

        self.fc2 = nn.Linear(hidden_sizes[0], hidden_sizes[1])
        self.dropout2 = nn.Dropout(dropout_rate)

        self.fc3 = nn.Linear(hidden_sizes[1], hidden_sizes[2])
        self.dropout3 = nn.Dropout(dropout_rate)

        self.fc4 = nn.Linear(hidden_sizes[2], num_classes)

        self._initialize_weights()

    def _initialize_weights(self):
        """Xavier initialization for better convergence."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.constant_(module.bias, 0)

    def forward(self, x):
        x = self.flatten(x)

        x = F.relu(self.fc1(x))
        x = self.dropout1(x)

        x = F.relu(self.fc2(x))
        x = self.dropout2(x)

        x = F.relu(self.fc3(x))
        x = self.dropout3(x)

        x = self.fc4(x)
        return x
