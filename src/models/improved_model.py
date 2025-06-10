import torch
import torch.nn as nn
import torch.nn.functional as F


class ImprovedMNISTClassifier(nn.Module):
    """
    Enhanced MNIST Classifier with batch normalization and deeper architecture.

    Features:
    - 4 hidden layers with batch normalization
    - Advanced regularization techniques
    - Kaiming weight initialization for ReLU networks

    Architecture: 784 → 1024 → 512 → 256 → 128 → 10
    """

    def __init__(self):
        super(ImprovedMNISTClassifier, self).__init__()

        self.flatten = nn.Flatten()

        # Enhanced architecture with batch normalization
        self.fc1 = nn.Linear(784, 1024)
        self.bn1 = nn.BatchNorm1d(1024)
        self.dropout1 = nn.Dropout(0.4)

        self.fc2 = nn.Linear(1024, 512)
        self.bn2 = nn.BatchNorm1d(512)
        self.dropout2 = nn.Dropout(0.4)

        self.fc3 = nn.Linear(512, 256)
        self.bn3 = nn.BatchNorm1d(256)
        self.dropout3 = nn.Dropout(0.3)

        self.fc4 = nn.Linear(256, 128)
        self.bn4 = nn.BatchNorm1d(128)
        self.dropout4 = nn.Dropout(0.2)

        self.fc5 = nn.Linear(128, 10)

        self._initialize_weights()

    def _initialize_weights(self):
        """Kaiming initialization for ReLU networks."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_uniform_(module.weight, nonlinearity='relu')
                nn.init.constant_(module.bias, 0)

    def forward(self, x):
        x = self.flatten(x)

        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout1(x)

        x = F.relu(self.bn2(self.fc2(x)))
        x = self.dropout2(x)

        x = F.relu(self.bn3(self.fc3(x)))
        x = self.dropout3(x)

        x = F.relu(self.bn4(self.fc4(x)))
        x = self.dropout4(x)

        x = self.fc5(x)
        return x
