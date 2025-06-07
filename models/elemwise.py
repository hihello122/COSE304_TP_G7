import torch.nn as nn

__all__ = ['ZeroMeanTransform']


class ZeroMeanTransform(nn.Module):

    def __init__(self):
        nn.Module.__init__(self)

    def forward(self, x, logpx=None):
        x = x - .5 # 입력 값의 평균을 0으로 근사
        if logpx is None:
            return x
        return x, logpx # logdet는 0이므로 logpx 변경 없음

    def inverse(self, y, logpy=None):
        y = y + .5
        if logpy is None:
            return y
        return y, logpy