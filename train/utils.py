# train/utils.py

import torch
import random
import numpy as np

def set_seed(seed):
    """모든 라이브러리의 난수 시드 고정"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def count_parameters(model):
    """총 파라미터 수 출력"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
