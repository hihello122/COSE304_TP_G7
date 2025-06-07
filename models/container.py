import torch.nn as nn

__all__ = ['SequentialFlow']


class SequentialFlow(nn.Module):
    """A generalized nn.Sequential container for normalizing flows.
    """

    def __init__(self, layersList):
        super(SequentialFlow, self).__init__()
        self.chain = nn.ModuleList(layersList)

    def forward(self, x, logpx=None):
        if logpx is None:
            for i in range(len(self.chain)):
                x = self.chain[i](x)
            return x
        else:
            for i in range(len(self.chain)):
                x, logpx = self.chain[i](x, logpx)
            return x, logpx

    def inverse(self, y, logpy=None):
        if logpy is None:
            for i in range(len(self.chain) - 1, -1, -1):
                y = self.chain[i].inverse(y)
            return y
        else:
            for i in range(len(self.chain) - 1, -1, -1):
                y, logpy = self.chain[i].inverse(y, logpy)
            return y, logpy