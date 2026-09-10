import math
from typing import Optional, Callable

from torch import nn

from ...utils import pair


def computeConvTensorHW(OriRes, KernelSize=3, Stride=2, Padding=None, Dilation=1, **kwargs):
    OriH, OriW = pair(OriRes)
    
    if Padding is None:
        Padding = int((KernelSize - 1) // 2 * Dilation)
        
    OutH = math.floor((OriH + 2 * Padding - Dilation * (KernelSize - 1) - 1) / Stride + 1)
    OutW = math.floor((OriW + 2 * Padding - Dilation * (KernelSize - 1) - 1) / Stride + 1)
            
    return OutH, OutW


def getTensorHWbyStage(InputRes, Stage, **kwargs):
    OutH, OutW = pair(InputRes)
    for _ in range(Stage):
        OutH, OutW = computeConvTensorHW((OutH, OutW))
    return OutH, OutW


def initActLayer(ActLayer: Optional[Callable[..., nn.Module]]=None):
    if ActLayer is not None:
        if isinstance(list(ActLayer().named_modules())[0][1], (nn.GELU, nn.Sigmoid)):
            ActLayer = ActLayer()
        else:
            ActLayer = ActLayer(inplace=True)
    return ActLayer