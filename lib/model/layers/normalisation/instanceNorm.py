from typing import Optional

from torch import nn

from .baseNorm import BaseNorm
from .normRegister import NORM_LAYER_REGISTRY


@NORM_LAYER_REGISTRY.register("instance_norm_3d")
class InstanceNorm3d(BaseNorm, nn.InstanceNorm3d):
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
        nn.InstanceNorm3d.__init__(
            self,
            num_features=NumFeatures,
            eps=Eps,
            momentum=Momentum,
            affine=affine,
            track_running_stats=track_running_stats,
        )
        

@NORM_LAYER_REGISTRY.register("instance_norm_2d")
class InstanceNorm2d(BaseNorm, nn.InstanceNorm2d):
    """
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
        **kwargs
    ) -> None:
        BaseNorm.__init__(self, **kwargs)
        nn.InstanceNorm2d.__init__(
            self,
            num_features=NumFeatures,
            eps=Eps,
            momentum=Momentum,
            affine=affine,
            track_running_stats=track_running_stats,
        )


@NORM_LAYER_REGISTRY.register("instance_norm_1d")
class InstanceNorm1d(BaseNorm, nn.InstanceNorm1d):
    """
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
        nn.InstanceNorm1d.__init__(
            self,
            num_features=NumFeatures,
            eps=Eps,
            momentum=Momentum,
            affine=affine,
            track_running_stats=track_running_stats,
        )