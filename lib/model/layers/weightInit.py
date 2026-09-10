import math

from torch import nn


NormLayerTuple = (
    nn.BatchNorm1d,
    nn.BatchNorm2d,
    nn.BatchNorm3d,
    nn.SyncBatchNorm,
    nn.LayerNorm,
    nn.InstanceNorm1d,
    nn.InstanceNorm2d,
    nn.InstanceNorm3d,
    nn.GroupNorm,
)


def initWeight(Module):
    """ init conv, norm, and linear layers
    self.apply works on recursively apply a function in modules of nn.Module
    nn.Parameter is not a module of nn.Module, the following won't work:
    ## positional embedding
    if isinstance(Module, nn.Parameter):
        nn.init.trunc_normal_(Module, std=.02)
    """
    # init conv, norm, and linear layers
    ## empty module
    if Module is None:
        return
    ## convolution layer
    elif isinstance(Module, (
            nn.Conv1d, nn.Conv2d, nn.Conv3d, 
            nn.ConvTranspose1d, nn.ConvTranspose2d, nn.ConvTranspose3d,
        )):
        nn.init.kaiming_uniform_(Module.weight, a=math.sqrt(5))
        if Module.bias is not None:
            FanIn, _ = nn.init._calculate_fan_in_and_fan_out(Module.weight)
            if FanIn != 0:
                bound = 1 / math.sqrt(FanIn)
                nn.init.uniform_(Module.bias, -bound, bound)
    ## normalisation layer
    elif isinstance(Module, NormLayerTuple):
        if Module.weight is not None:
            nn.init.ones_(Module.weight)
        if Module.bias is not None:
            nn.init.zeros_(Module.bias)
    ## linear layer
    elif isinstance(Module, nn.Linear):
        nn.init.kaiming_uniform_(Module.weight, a=math.sqrt(5))
        if Module.bias is not None:
            FanIn, _ = nn.init._calculate_fan_in_and_fan_out(Module.weight)
            bound = 1 / math.sqrt(FanIn) if FanIn > 0 else 0
            nn.init.uniform_(Module.bias, -bound, bound)
    ## loop inside
    elif isinstance(Module, (nn.Sequential, nn.ModuleList)):
        for m in Module:
            initWeight(m)
    ## loop inside
    elif list(Module.children()):
        for m in Module.children():
            initWeight(m)
