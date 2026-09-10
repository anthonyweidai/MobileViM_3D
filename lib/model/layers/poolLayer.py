from typing import Union

import torch
from torch import nn, Tensor
from torch.nn import functional as F
from torch.nn.modules.pooling import _MaxPoolNd, _AvgPoolNd, _AdaptiveAvgPoolNd

from ...utils import pair


class MaxPool(_MaxPoolNd):
    def __init__(
        self, KernelSize: int, Stride: int=None, 
        Padding: int=0, Dilation: int= 1, SpatialDims: int=2, 
        ReturnIndices: bool=False, CeilMode: bool=False,
    ) -> None:
        KernelSize = pair(KernelSize, RepNum=SpatialDims)
        Stride = pair(Stride, RepNum=SpatialDims)
        Padding = pair(Padding, RepNum=SpatialDims)
        Dilation = pair(Dilation, RepNum=SpatialDims)
        super().__init__(
            kernel_size=KernelSize, stride=Stride, padding=Padding, dilation=Dilation,
            return_indices=ReturnIndices, ceil_mode=CeilMode,
        )
        self.SpatialDims = SpatialDims

    def forward(self, Input: Tensor):
        return F.max_pool3d(
            Input, self.kernel_size, self.stride, self.padding, self.dilation, 
            ceil_mode=self.ceil_mode, return_indices=self.return_indices,
        )


class AvgPool(_AvgPoolNd):
    def __init__(
        self, KernelSize: int, Stride: int=None, 
        Padding: int=0, Dilation: int= 1, SpatialDims: int=2, 
        CeilMode: bool=False, CountIncludePad: bool=True, 
        DivisorOverride: int=None,
    ) -> None:
        super().__init__()
        self.CeilMode = CeilMode
        self.CountIncludePad = CountIncludePad
        self.DivisorOverride = DivisorOverride
        
        self.KernelSize = pair(KernelSize, RepNum=3)
        self.Stride = pair(Stride, RepNum=3)
        self.Padding = pair(Padding, RepNum=3)
        self.Dilation = pair(Dilation, RepNum=3)
        
    def forward(self, Input: Tensor):
        return F.avg_pool3d(
            Input, self.KernelSize, self.Stride, self.Padding, self.Dilation, 
            ceil_mode=self.CeilMode, count_include_pad=self.CountIncludePad,
            divisor_override=self.DivisorOverride,
        )


class AdaptiveAvgPool(_AdaptiveAvgPoolNd):
    def __init__(self, OutputSize: Union[int, tuple], SpatialDims: int=2):
        OutputSize = pair(OutputSize, RepNum=SpatialDims)
        super(AdaptiveAvgPool, self).__init__(output_size=OutputSize)
        self.SpatialDims = SpatialDims
         
    def forward(self, Input: Tensor) -> Tensor:
        return F.adaptive_avg_pool3d(Input, self.output_size)
    

class Globalpooling(nn.Module):
    def __init__(self, PoolType="mean", KeepDim=False, SpatialDims: int=2):
        super(Globalpooling, self).__init__()
        self.PoolType = PoolType
        self.KeepDim = KeepDim
        
        self.Dims = [- i for i in reversed(range(1, 4))]
        
    def globalPool(self, x):
        if self.PoolType == "rms":
            x = x ** 2
            x = torch.mean(x, dim=self.Dims, keepdim=self.KeepDim)
            x = x ** -0.5
        elif self.PoolType == "abs":
            x = torch.mean(x, dim=self.Dims, keepdim=self.KeepDim)
        else:
            # same as AdaptiveAvgPool
            x = torch.mean(torch.abs(x), dim=self.Dims, keepdim=self.KeepDim)# use default method "mean"
        
        return x
        
    def forward(self, x: Tensor) -> Tensor:
        return self.globalPool(x)