import types
from typing import Optional, Callable

from torch import nn, Tensor

from .utils import initActLayer
from .weightInit import initWeight
from .normalisation import buildNormLayer


class BaseConv(nn.Module):
    def __init__(self, opt, Bias, NormType=None, UseNorm=True, **kwargs) -> None:
        super().__init__()
        if not UseNorm:
            NormType = None
        elif NormType is None:
            NormType = opt.norm_layer
        
        self.opt = self.getConvOptions(opt, NormType)
        
        if Bias is None:
            self.Bias = NormType is None
        else:
            self.Bias = Bias
    
    @staticmethod
    def getConvOptions(opt, NormType):
        Options = types.SimpleNamespace()
        setattr(Options, "spatial_dims", opt.spatial_dims)
        setattr(Options, "norm_layer", NormType)
        setattr(Options, "device", opt.device)
        return Options


class Convolution(BaseConv):
    def __init__(
        self,
        opt,
        InChannels: int,
        OutChannels: int,
        KernelSize: int,
        Stride: Optional[int]=1,
        Padding: Optional[int]=None,
        Dilation: Optional[int]=1,
        Groups: Optional[int]=1,
        Bias: Optional[bool]=None,
        ActLayer: Optional[Callable[..., nn.Module]]=None,
        Momentum: Optional[float]=0.1,
        NormType: str=None,
        UseNorm: bool=True,
        **kwargs,
    ) -> None:
        super().__init__(opt, Bias, NormType, UseNorm, **kwargs)
        # support 2/3 D convolutions
        if Padding is None:
            Padding = int((KernelSize - 1) // 2 * Dilation)
            
        self.InChannels = InChannels
        self.OutChannels = OutChannels
        self.KernelSize = KernelSize
        self.Stride = Stride
        self.Padding = Padding
        self.Groups = Groups
        
        Conv = nn.Conv3d
        self.Conv = Conv(
            in_channels=InChannels, 
            out_channels=OutChannels, 
            kernel_size=KernelSize, 
            stride=Stride, 
            padding=Padding, 
            dilation=Dilation, 
            groups=Groups, 
            bias=self.Bias, 
            **kwargs,
        )
        
        self.Bn = buildNormLayer(self.opt)(
            NumFeatures=OutChannels, Eps=0.001, Momentum=Momentum, **kwargs,
        ) if self.opt.norm_layer is not None else nn.Identity()
        
        self.Act = initActLayer(ActLayer)
        
        self.apply(initWeight)
        
    def forward(self, x: Tensor) -> Tensor:
        x = self.Conv(x)
        x = self.Bn(x)
        if self.Act is not None:
            x = self.Act(x)
        return x


class TransposeConvLayer(BaseConv):
    def __init__(
        self,
        opt,
        InChannels: int,
        OutChannels: int,
        KernelSize: int,
        Stride: Optional[int] = 1,
        Padding: Optional[int]=None,
        OutputPadding: Optional[int] = None,
        Dilation: Optional[int] = 1,
        Groups: Optional[int] = 1,
        Bias: Optional[bool]=None,
        PaddingMode: Optional[str] = "zeros",
        ActLayer: Optional[Callable[..., nn.Module]]=None,
        Momentum: Optional[float]=0.1,
        NormType: str=None,
        UseNorm: bool=True,
        **kwargs,
    ):
        super().__init__(opt, Bias, NormType, UseNorm, **kwargs)
        if Padding is None:
            Padding = int((KernelSize - 1) // 2 * Dilation)
        
        if OutputPadding is None:
            OutputPadding = Stride - 1
            
        self.InChannels = InChannels
        self.OutChannels = OutChannels
        self.KernelSize = KernelSize
        self.Stride = Stride
        self.Padding = Padding
        self.Groups = Groups
        
        Conv = nn.ConvTranspose3d
        self.Conv = Conv(
            in_channels=InChannels, 
            out_channels=OutChannels, 
            kernel_size=KernelSize, 
            stride=Stride,
            padding=Padding, 
            output_padding=OutputPadding, 
            groups=Groups, 
            bias=self.Bias, 
            dilation=Dilation, 
            padding_mode=PaddingMode,
            **kwargs,
        )
        
        self.Bn = buildNormLayer(self.opt)(
            NumFeatures=OutChannels, Eps=0.001, Momentum=Momentum, **kwargs,
        ) if self.opt.norm_layer is not None else nn.Identity()
        
        self.Act = initActLayer(ActLayer)
        
        self.apply(initWeight)

    def forward(self, x: Tensor) -> Tensor:
        x = self.Conv(x)
        x = self.Bn(x)
        if self.Act is not None:
            x = self.Act(x)
        return x
