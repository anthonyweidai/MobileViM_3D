from typing import Optional, Union, List

import torch
from torch import nn, Tensor
import torch.nn.functional as F

from .baseNorm import BaseNorm
from .normRegister import NORM_LAYER_REGISTRY
from ..weightInit import initWeight


@NORM_LAYER_REGISTRY.register("layer_norm")
class LayerNorm(BaseNorm, nn.LayerNorm):
    """ The ordering of the dimensions in the inputs. 
    Inputs with shape (batch_size, height, width, channels).
    The reason to sync BatchNorm is because it collects statistics across samples 
    (i.e. elements of a minibatch) which will be on different GPUs.
    LayerNorm does not merge statistics between elements of a minibatch but only computes statistics within a sample, 
    which will be on a given GPU. So there is nothing to sync.
    https://discuss.pytorch.org/t/layer-norm-sync-during-distributed-training/138312/2?u=anthony_dave
    """
    def __init__(
        self, NumFeatures, Eps: float = 1e-5, Bias=True, 
        Device=None, DType=None, **kwargs,
    ) -> None:
        factory_kwargs = {'device': Device, 'dtype': DType}
        BaseNorm.__init__(self, **kwargs)
        nn.LayerNorm.__init__(self, normalized_shape=NumFeatures, eps=Eps, device=Device, dtype=DType, **kwargs)
        if self.elementwise_affine:
            self.weight = nn.Parameter(torch.empty(self.normalized_shape, **factory_kwargs))
            if Bias:
                self.bias = nn.Parameter(torch.empty(self.normalized_shape, **factory_kwargs))
            else:
                self.register_parameter('bias', None)
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        if self.elementwise_affine:
            nn.init.ones_(self.weight)
            if self.bias is not None:
                nn.init.zeros_(self.bias)
                
                
@NORM_LAYER_REGISTRY.register("layer_norm_fp32")
class LayerNormFP32(LayerNorm):
    """
    Applies `Layer Normalization <https://arxiv.org/abs/1607.06450>`_ over a input tensor with FP32 precision
    """

    def __init__(
        self,
        NumFeatures: Union[int, List[int], torch.Size],
        Eps: Optional[float] = 1e-5,
        elementwise_affine: Optional[bool] = True,
        **kwargs
    ):
        super().__init__(
            NumFeatures=NumFeatures,
            Eps=Eps,
            elementwise_affine=elementwise_affine,
            **kwargs
        )

    def forward(self, x: Tensor) -> Tensor:
        # Convert input from dtype X to FP32 and perform normalization operation.
        # This may help with underflow/overflow issues that we typically see with normalization layers
        inp_dtype = x.dtype
        return super().forward(x.to(torch.float32)).to(inp_dtype)


@NORM_LAYER_REGISTRY.register("layer_norm_cf")
class LayerNormChannelFirst(BaseNorm):
    """ The ordering of the dimensions in the inputs. channels_first corresponds to inputs 
    with shape (batch_size, channels, height, width).
    """
    def __init__(self, NumFeatures, Eps=1e-6, **kwargs):
        BaseNorm.__init__(self, **kwargs)
        nn.Module.__init__(self)
        self.Eps = Eps
        self.NumFeatures = (NumFeatures, )
        
        self.weight = nn.Parameter(torch.ones(NumFeatures))
        self.bias = nn.Parameter(torch.zeros(NumFeatures))
        
    def forward(self, x: Tensor) -> Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.Eps)
        x = self.weight[:, None, None, None] * x + self.bias[:, None, None, None]
        # python >= 3.11
        # x = self.weight[:, *([None] * SpatialDims)] * x + self.bias[:, *([None] * SpatialDims)]
        return x


@NORM_LAYER_REGISTRY.register("dual_layer_norm")
class DualLayernorm(BaseNorm):
    # 3.1 Layernorm(LN) is applied before every block
    def __init__(self, DimEmb, Fn=None, **kwargs):
        super().__init__(**kwargs)
        self.LayerNorm = nn.LayerNorm(DimEmb)
        self.Fn = Fn
        
        self.apply(initWeight)
        
    def forward(self, x, Feature, **kwargs):
        x = self.LayerNorm(x)
        if self.Fn:
            x = self.Fn(x, **kwargs)
            
        Feature = self.LayerNorm(Feature)
        if self.Fn:
            Feature = self.Fn(Feature, **kwargs)
        return x, Feature