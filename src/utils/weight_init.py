import torch
from torch import nn


def kaiming_uniform(module: nn.Module) -> None:
    """
    Initialization for modules using relu.
    """
    nn.init.kaiming_uniform_(module.weight, mode="fan_in", nonlinearity="relu")
    if module.bias is not None:
        nn.init.constant_(module.bias, 0.01)

        
def xavier_uniform(module: nn.Module) -> None:
    """
    Initialization for modules without activation function.
    """
    nn.init.xavier_uniform_(module.weight, gain=1.0)
    nn.init.xavier_uniform_(module.bias, gain=1.0)


def c2_xavier_fill(module: nn.Module) -> None:
    """
    Initialize `module.weight` using the "XavierFill" implemented in Caffe2.
    Also initializes `module.bias` to 0.
    Args:
        module (torch.nn.Module): module to initialize.
    """
    # Caffe2 implementation of XavierFill in fact
    # corresponds to kaiming_uniform_ in PyTorch
    nn.init.kaiming_uniform_(module.weight, a=1)  # pyre-ignore
    if module.bias is not None:  # pyre-ignore
        nn.init.constant_(module.bias, 0)


def c2_msra_fill(module: nn.Module) -> None:
    """
    Initialize `module.weight` using the "MSRAFill" implemented in Caffe2.
    Also initializes `module.bias` to 0.
    Args:
        module (torch.nn.Module): module to initialize.
    """
    # pyre-ignore
    nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
    if module.bias is not None:  # pyre-ignore
        nn.init.constant_(module.bias, 0)