# models/residual_block.py

import torch
import torch.nn as nn
from torch.autograd import grad

class ResidualBlock(nn.Module):
    def __init__(self, hidden_channels):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, hidden_channels)
        )

    def forward(self, x):
        # Residual connection
        Fx = self.net(x)
        z = x + Fx

        # Hutchinson's trace estimator
        v = torch.randn_like(x)
        Jv = grad(Fx, x, v, retain_graph=True, create_graph=True)[0]
        trace_est = (v * Jv).sum(dim=1)

        return z, trace_est
