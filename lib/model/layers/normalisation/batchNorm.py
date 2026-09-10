from typing import Optional

import torch
from torch import nn, Tensor

from .baseNorm import BaseNorm
from .normRegister import NORM_LAYER_REGISTRY


@NORM_LAYER_REGISTRY.register("batch_norm_3d")
class BatchNorm3d(BaseNorm, nn.BatchNorm3d):
    def __init__(
        self,
        NumFeatures: int,
        Eps: Optional[float] = 1e-5,
        Momentum: Optional[float] = 0.1,
        affine: Optional[bool] = True,
        track_running_stats: Optional[bool] = True,
        **kwargs
    ) -> None:
        """
        Applies a `Batch Normalization <https://arxiv.org/abs/1502.03167>`_ over a 5D input tensor

        Args:
            NumFeatures (Optional, int): :math:`C` from an expected input of size :math:`(N, C, D, H, W)`
            Eps (Optional, float): Value added to the denominator for numerical stability. Default: 1e-5
            Momentum (Optional, float): Value used for the running_mean and running_var computation. Default: 0.1
            affine (bool): If ``True``, use learnable affine parameters. Default: ``True``
            track_running_stats: If ``True``, tracks running mean and variance. Default: ``True``

        Shape:
            - Input: :math:`(N, C, D, H, W)` where :math:`N` is the batch size, :math:`C` is the number of input
            channels, :math:`D` is the input depth, :math:`H` is the input height, and :math:`W` is the input width
            - Output: same shape as the input
        """
        BaseNorm.__init__(self, **kwargs)
        nn.BatchNorm3d.__init__(
            self,
            num_features=NumFeatures,
            eps=Eps,
            momentum=Momentum,
            affine=affine,
            track_running_stats=track_running_stats,
        )
        

@NORM_LAYER_REGISTRY.register("batch_norm_3d_fp32")
class BatchNorm3dFP32(BaseNorm, nn.BatchNorm3d):
    def __init__(
        self,
        NumFeatures: int,
        Eps: Optional[float] = 1e-5,
        Momentum: Optional[float] = 0.1,
        affine: Optional[bool] = True,
        track_running_stats: Optional[bool] = True,
        **kwargs
    ) -> None:
        """
        Applies a `Batch Normalization <https://arxiv.org/abs/1502.03167>`_ over a 5D input tensor

        Args:
            NumFeatures (Optional, int): :math:`C` from an expected input of size :math:`(N, C, D, H, W)`
            Eps (Optional, float): Value added to the denominator for numerical stability. Default: 1e-5
            Momentum (Optional, float): Value used for the running_mean and running_var computation. Default: 0.1
            affine (bool): If ``True``, use learnable affine parameters. Default: ``True``
            track_running_stats: If ``True``, tracks running mean and variance. Default: ``True``

        Shape:
            - Input: :math:`(N, C, D, H, W)` where :math:`N` is the batch size, :math:`C` is the number of input
            channels, :math:`D` is the input depth, :math:`H` is the input height, and :math:`W` is the input width
            - Output: same shape as the input
        """
        BaseNorm.__init__(self, **kwargs)
        nn.BatchNorm3d.__init__(
            self,
            num_features=NumFeatures,
            eps=Eps,
            momentum=Momentum,
            affine=affine,
            track_running_stats=track_running_stats,
        )
    
    def forward(self, input: Tensor) -> Tensor:
        return super().forward(input.to(torch.float32)).to(input.dtype)


@NORM_LAYER_REGISTRY.register("batch_norm_2d")
class BatchNorm2d(BaseNorm, nn.BatchNorm2d):
    """
    Applies a `Batch Normalization <https://arxiv.org/abs/1502.03167>`_ over a 4D input tensor

    Args:
        NumFeatures (Optional, int): :math:`C` from an expected input of size :math:`(N, C, H, W)`
        Eps (Optional, float): Value added to the denominator for numerical stability. Default: 1e-5
        Momentum (Optional, float): Value used for the running_mean and running_var computation. Default: 0.1
        affine (bool): If ``True``, use learnable affine parameters. Default: ``True``
        track_running_stats: If ``True``, tracks running mean and variance. Default: ``True``

    Shape:
        - Input: :math:`(N, C, H, W)` where :math:`N` is the batch size, :math:`C` is the number of input channels,
        :math:`H` is the input height, and :math:`W` is the input width
        - Output: same shape as the input
    """

    def __init__(
        self,
        NumFeatures: int,
        Eps: Optional[float] = 1e-5,
        Momentum: Optional[float] = 0.1,
        affine: Optional[bool] = True,
        track_running_stats: Optional[bool] = True,
        **kwargs,
    ) -> None:
        BaseNorm.__init__(self, **kwargs)
        nn.BatchNorm2d.__init__(
            self,
            num_features=NumFeatures,
            eps=Eps,
            momentum=Momentum,
            affine=affine,
            track_running_stats=track_running_stats,
        )


@NORM_LAYER_REGISTRY.register("batch_norm_2d_fp32")
class BatchNorm2dFP32(BatchNorm2d):
    """
    Applies a `Batch Normalization <https://arxiv.org/abs/1502.03167>`_ over a 4D input tensor in FP32
    """

    def __init__(
        self,
        NumFeatures: int,
        Eps: Optional[float] = 1e-5,
        Momentum: Optional[float] = 0.1,
        affine: Optional[bool] = True,
        track_running_stats: Optional[bool] = True,
        **kwargs
    ) -> None:
        super().__init__(
            NumFeatures=NumFeatures,
            Eps=Eps,
            Momentum=Momentum,
            affine=affine,
            track_running_stats=track_running_stats,
            **kwargs
        )

    def forward(self, input: Tensor) -> Tensor:
        return super().forward(input.to(torch.float32)).to(input.dtype)


@NORM_LAYER_REGISTRY.register("batch_norm_1d")
class BatchNorm1d(BaseNorm, nn.BatchNorm1d):
    """
    Applies a `Batch Normalization <https://arxiv.org/abs/1502.03167>`_ over a 2D or 3D input tensor

    Args:
        NumFeatures (Optional, int): :math:`C` from an expected input of size :math:`(N, C)` or :math:`(N, C, L)`
        Eps (Optional, float): Value added to the denominator for numerical stability. Default: 1e-5
        Momentum (Optional, float): Value used for the running_mean and running_var computation. Default: 0.1
        affine (bool): If ``True``, use learnable affine parameters. Default: ``True``
        track_running_stats: If ``True``, tracks running mean and variance. Default: ``True``

    Shape:
        - Input: :math:`(N, C)` or :math:`(N, C, L)` where :math:`N` is the batch size,
        :math:`C` is the number of input channels,  and :math:`L` is the sequence length
        - Output: same shape as the input
    """

    def __init__(
        self,
        NumFeatures: int,
        Eps: Optional[float] = 1e-5,
        Momentum: Optional[float] = 0.1,
        affine: Optional[bool] = True,
        track_running_stats: Optional[bool] = True,
        **kwargs
    ) -> None:
        BaseNorm.__init__(self, **kwargs)
        nn.BatchNorm1d.__init__(
            self,
            num_features=NumFeatures,
            eps=Eps,
            momentum=Momentum,
            affine=affine,
            track_running_stats=track_running_stats,
        )


@NORM_LAYER_REGISTRY.register("batch_norm_1d_fp32")
class BatchNorm1dFP32(BaseNorm, nn.BatchNorm1d):
    """
    Applies a `Batch Normalization <https://arxiv.org/abs/1502.03167>`_ over a 2D or 3D input tensor

    Args:
        NumFeatures (Optional, int): :math:`C` from an expected input of size :math:`(N, C)` or :math:`(N, C, L)`
        Eps (Optional, float): Value added to the denominator for numerical stability. Default: 1e-5
        Momentum (Optional, float): Value used for the running_mean and running_var computation. Default: 0.1
        affine (bool): If ``True``, use learnable affine parameters. Default: ``True``
        track_running_stats: If ``True``, tracks running mean and variance. Default: ``True``

    Shape:
        - Input: :math:`(N, C)` or :math:`(N, C, L)` where :math:`N` is the batch size,
        :math:`C` is the number of input channels,  and :math:`L` is the sequence length
        - Output: same shape as the input
    """

    def __init__(
        self,
        NumFeatures: int,
        Eps: Optional[float] = 1e-5,
        Momentum: Optional[float] = 0.1,
        affine: Optional[bool] = True,
        track_running_stats: Optional[bool] = True,
        **kwargs
    ) -> None:
        BaseNorm.__init__(self, **kwargs)
        nn.BatchNorm1d.__init__(
            self,
            num_features=NumFeatures,
            eps=Eps,
            momentum=Momentum,
            affine=affine,
            track_running_stats=track_running_stats,
        )
    
    def forward(self, input: Tensor) -> Tensor:
        return super().forward(input.to(torch.float32)).to(input.dtype)   


@NORM_LAYER_REGISTRY.register("batch_norm_1d_no_biase")
class BatchNorm1dNoBias(BaseNorm, nn.BatchNorm1d):
    def __init__(self, **kwargs):
        BaseNorm.__init__(self, **kwargs)
        nn.BatchNorm1d.__init__(self, **kwargs)
        self.bias.requires_grad = False