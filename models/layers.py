# models/layers.py

import torch
import torch.nn as nn

class ActNorm(nn.Module):
    """ Activation Normalization (learnable per-channel affine) """
    def __init__(self, num_features):
        super().__init__()
        self.initialized = False
        self.scale = nn.Parameter(torch.ones(1, num_features, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, num_features, 1, 1))

    def initialize(self, x):
        # x shape: (B, C, H, W)
        with torch.no_grad():
            mean = x.mean(dim=[0, 2, 3], keepdim=True)
            std = x.std(dim=[0, 2, 3], keepdim=True)
            self.bias.data.copy_(-mean)
            self.scale.data.copy_(1 / (std + 1e-6))
        self.initialized = True

    def forward(self, x):
        if not self.initialized:
            self.initialize(x)
        log_abs = torch.log(torch.abs(self.scale) + 1e-6).sum()
        log_det = x.size(2) * x.size(3) * log_abs
        return self.scale * (x + self.bias), log_det
