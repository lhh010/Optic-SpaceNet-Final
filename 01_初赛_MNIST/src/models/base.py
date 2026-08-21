"""Full-precision photonic MLP base model.

References:
    - src_raw/train_with_local_data.py
    - src_raw/train_with_api.py
"""

import torch
import torch.nn as nn


class PhotonicMLP(nn.Module):
    """3-layer MLP for photonic MNIST classification."""

    def __init__(self, hidden_dim1: int = 128, hidden_dim2: int = 64):
        super().__init__()
        self.fc1 = nn.Linear(784, hidden_dim1, bias=False)
        self.fc2 = nn.Linear(hidden_dim1, hidden_dim2, bias=False)
        self.fc3 = nn.Linear(hidden_dim2, 10, bias=False)

    def forward(self, x):
        x = x.view(-1, 784)
        h1 = torch.relu(self.fc1(x))
        h2 = torch.relu(self.fc2(h1))
        out = self.fc3(h2)
        return out
