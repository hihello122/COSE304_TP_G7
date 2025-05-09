# models/flow.py

import torch.nn as nn
from models.residual_block import ResidualBlock

class ResidualFlow(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.blocks = nn.ModuleList([
            ResidualBlock(config['hidden_channels']) for _ in range(config['num_blocks'])
        ])

    def forward(self, x):
        log_det = 0
        for block in self.blocks:
            x, ld = block(x)
            log_det += ld
        return x, log_det
