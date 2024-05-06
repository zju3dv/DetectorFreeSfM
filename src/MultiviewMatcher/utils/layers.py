from functools import partial

import torch.nn as nn
from torch.nn.modules.batchnorm import _BatchNorm


def build_norm(norm, nc, l0=None, l1=None):
    if issubclass(norm, nn.LayerNorm):
        n_shape = list(filter(lambda x: x is not None, [nc, l0, l1]))
        return partial(norm, n_shape, elementwise_affine=True)
    elif issubclass(norm, _BatchNorm):
        return partial(norm, nc, affine=True)
    else:
        raise NotImplementedError()

def fc_norm_relu(nc_in, nc_out, norm=nn.LayerNorm, relu=nn.ReLU):
    bias = False if issubclass(norm, _BatchNorm) else True
    norm = build_norm(norm, nc_out)
    return nn.Sequential(
        nn.Linear(nc_in, nc_out, bias=bias), norm(), relu())

class Lambda(nn.Module):
    "An easy way to create a pytorch layer for a simple `func`."
    def __init__(self, func):
        super().__init__()
        self.func=func

    def forward(self, x): 
        return self.func(x)
