import re
import numpy as np
from einops import rearrange
from functools import partial
from typing import Tuple, Optional, Dict

import torch
from torch import nn, Tensor
from timm.models.layers import DropPath 

try:
    from mamba_ssm.ops.triton.layernorm import RMSNorm, layer_norm_fn, rms_norm_fn
except ImportError:
    RMSNorm, layer_norm_fn, rms_norm_fn = None, None, None

from .mamba import Mamba2
from ..layers import Convolution, AdaptiveAvgPool, StochasticDepth
from ...utils import pair, setMethod, callMethod, unsqueezeRight


class MambaBlock(nn.Module):
    """ Simple block wrapping a mixer class with LayerNorm/RMSNorm and residual connection"
    This Block has a slightly different structure compared to a regular
    prenorm Transformer block.
    The standard block is: LN -> MHA/MLP -> Add.
    [Ref: https://arxiv.org/abs/2002.04745]
    """
    def __init__(
        self, opt, Dim, Mamba, MambaNorm=nn.LayerNorm, DropRate=0.,
    ):
        super().__init__()
        self.opt = opt
        self.ResidualFP32 = opt.use_fp32_residual

        self.Mamba = Mamba(Dim)
        self.Norm = MambaNorm(Dim)
        self.DropPath = DropPath(DropRate) if DropRate > 0. else nn.Identity()
        self.FusedAddNormFn = rms_norm_fn if isinstance(self.Norm, RMSNorm) else layer_norm_fn
        assert isinstance(self.Norm, (nn.LayerNorm, RMSNorm)), \
            "Only LayerNorm and RMSNorm are supported for FusedAddNorm"

    def forward(self, Input: dict):
        r"""Pass the input through the encoder layer.

        Args:
            HiddenStates: the sequence to the encoder layer (required).
            Residual: HiddenStates = Mamba(LN(residual))
        """
        HiddenStates, Residual = Input["hidden_states"], Input["residual"]
        
        if Residual is None:
            HiddenStates, Residual = self.FusedAddNormFn(
                HiddenStates,
                self.Norm.weight,
                self.Norm.bias,
                residual=Residual,
                prenorm=True,
                residual_in_fp32=self.ResidualFP32,
                eps=self.Norm.eps,
            )
        else:
            HiddenStates, Residual = self.FusedAddNormFn(
                self.DropPath(HiddenStates),
                self.Norm.weight,
                self.Norm.bias,
                residual=Residual,
                prenorm=True,
                residual_in_fp32=self.ResidualFP32,
                eps=self.Norm.eps,
            )
        
        HiddenStates = self.Mamba(HiddenStates)
            
        return {
            "hidden_states": HiddenStates,
            "residual": Residual,
        }


class BaseMambaBlock(nn.Module):
    def __init__(
        self, 
        opt,
        UseMamba,
        DimEmb: int, 
        PatchSize: int, 
        NumMambaBlocks: int=2, 
        DropRate: Optional[float]=0., 
        FuseDropRate: Optional[float]=0., 
        **kwargs,
        ):
        super().__init__()
        self.opt = opt
        self.UseMamba = UseMamba
        
        self.SpatialDims = opt.spatial_dims
        self.PatchLens = pair(PatchSize, RepNum=opt.spatial_dims)
        self.PatchArea = np.prod(self.PatchLens)
        self.DimEmb = DimEmb * self.PatchArea # C x P
        
        if UseMamba:
            self.initRearrangePattern()
            
            MambaDimEmb = self.DimEmb
            if opt.mamba_dimin == 1:
                # assume that patch lengths are the same among axes
                # expand ratio: PatchArea // PatchLens[0]
                if opt.mamba_dimin_expand:
                    OutChannels = self.DimEmb // self.PatchLens[0]
                else:
                    MambaDimEmb = DimEmb * self.PatchLens[0]
                    
                for i in range(opt.spatial_dims):
                    OutputSize = [1] * opt.spatial_dims
                    OutputSize[i] = None
                    Layer = nn.Sequential(
                        AdaptiveAvgPool(OutputSize, SpatialDims=opt.spatial_dims),
                        Convolution(opt, DimEmb, OutChannels, 1, 1, ActLayer=nn.SiLU) \
                            if opt.mamba_dimin_expand else nn.Identity(),
                    )
                    setMethod(self, "InDPConv%d" % (i + 1), Layer)
                
                if opt.mamba_dimin_expand:
                    self.OutDPConv = Convolution(opt, OutChannels, DimEmb, 1, 1, ActLayer=nn.SiLU)
            
            GlobalRep = nn.ModuleList()
            for i in range(NumMambaBlocks):
                GlobalRep.append(nn.Identity())
                MambaModule = partial(
                    Mamba2, 
                    d_state=opt.d_state, layer_idx=i, 
                    DualDirection=opt.mamba_dualdirection, IsCatProj=opt.mamba_catproj,
                )
                MambaNorm = partial(RMSNorm, eps=1e-7)
                GlobalRep.append(MambaBlock(
                    opt, MambaDimEmb, Mamba=MambaModule, MambaNorm=MambaNorm, DropRate=DropRate,
                ))
            self.GlobalRep = nn.Sequential(*GlobalRep)
        
        self.Dropout = StochasticDepth(FuseDropRate)
    
    def initRearrangePattern(self):
        # rearrangement pattern
        SpatialDims = self.SpatialDims if self.opt.mamba_dimin != 1 else 1
        
        SourcePattern = ["(n%d p%d)" % (i + 1, i + 1) for i in range(SpatialDims)]
        TargetPattern1 = ["n%d" % (i + 1) for i in range(SpatialDims)]
        TargetPattern2 = ["p%d" % (i + 1) for i in range(SpatialDims)]
        SourcePattern = ' '.join(SourcePattern)
        TargetPattern1 = ' '.join(TargetPattern1)
        TargetPattern2 = ' '.join(TargetPattern2)
        self.UnfoldPattern = "b c %s -> b (%s) (c %s)" % (
            SourcePattern, TargetPattern1, TargetPattern2,
        )
        self.FoldPattern = "b (%s) (c %s) -> b c %s" % (
            TargetPattern1, TargetPattern2, SourcePattern,
        )
        
        # axis lengths, any additional specifications for dimensions
        ALDict = {} 
        for i, p in enumerate(self.PatchLens):
            ALDict.update({"p%d" % (i + 1): p})
            # assume that patch lengths are the same among axes
            if self.opt.mamba_dimin == 1: break
        self.ALDict = ALDict
        
    def unfolding(self, FeatureMap: Tensor) -> Tuple[Tensor, Tuple[int, int]]:
        # arrange shape
        SpatialDims = self.SpatialDims if self.opt.mamba_dimin != 1 else 1
        
        # [B, C, *] --> [B, C, n_1, p_1, n_2, p_2, n_3, p_3, ...]
        # [B, C, n_1, p_1, n_2, p_2, n_3, p_3, ...] --> [B, C x P, N]
        # [B, C x P, N] --> [B, N, C x P]
        # N: number of patches, P: patch length
        Patches = rearrange(FeatureMap, self.UnfoldPattern, **self.ALDict)
        
        # information dictionary
        NumPatches = (
            FeatureMap.shape[-SpatialDims:] / np.array(self.PatchLens)
        ).astype(int)
        InfoDict = {}
        for i, n in enumerate(NumPatches):
            InfoDict.update({"n%d" % (i + 1): n})
            # assume that patch lengths are the same among axes
            if self.opt.mamba_dimin == 1: break
        
        return Patches, InfoDict

    def folding(self, Patches: Tensor, InfoDict: dict) -> Tensor:
        # [B, N, C x P] --> [B, C x P, N]
        # [B, C x P, N] --> [B, C, n_1, p_1, n_2, p_2, n_3, p_3, ...]
        # [B, C, n_1, p_1, n_2, p_2, n_3, p_3, ...] --> [B, C, *]
        return rearrange(Patches, self.FoldPattern, **(self.ALDict | InfoDict))

    def mambaForward(self, FeatureMap: Tensor) -> Tensor:
        if self.UseMamba:
            # list input for looping
            Input = []
            if self.opt.mamba_dimin == 1:
                # dimension-independent convolution
                SqueezeList = list(range(self.SpatialDims))
                for i in SqueezeList:
                    DPFeatures = callMethod(self, "InDPConv%d" % (i + 1))(FeatureMap)
                    for j in reversed(SqueezeList):
                        DPFeatures.squeeze_(-(1 + j))
                    Input.append(DPFeatures)
            else:
                Input = [FeatureMap]
            
            MambaOutput = []
            for f in Input:
                # convert feature maps to patches
                # Patches = DPatch x HPatch x WPatch, N is sequence length
                # [B x C // 2 x D x H x W] --> [B x C // 2  x Patches x N]
                Patches, InfoDict = self.unfolding(f)
                # learn global representations on all patches
                MambaInput = {"hidden_states": Patches, "residual": None}
                # [B x C // 2  x Patches x N] --> [B x C // 2 x Patches x N]
                Patches = self.GlobalRep(MambaInput)["hidden_states"]
                # convert patches to feature maps
                # [B x C // 2  x Patches x N] --> [B x C // 2 x D x H x W]
                MambaOutput.append(self.folding(Patches, InfoDict))
            
            LenMambaOut = len(MambaOutput)
            if LenMambaOut > 1:
                MambaOutput = [
                    unsqueezeRight(o, 1).movedim(-2, -(2 - min(i, 1))) \
                        for i, o in enumerate(MambaOutput)
                ]
                if LenMambaOut == 3:
                    Output12 = torch.matmul(MambaOutput[0], MambaOutput[1])
                    Output13 = torch.matmul(MambaOutput[0], MambaOutput[2])
                    Output = torch.matmul(Output12.unsqueeze(-1), Output13.unsqueeze(-2))
                    if self.opt.mamba_dimin_expand: Output = self.OutDPConv(Output)
                else:
                    raise NotImplementedError
                
            else:
                Output = MambaOutput[0]
                
            return Output
        else:
            return FeatureMap
    
    def forward(self, x: Tensor) -> Tensor:
        raise NotImplementedError


class MobileMambaBlockv1(BaseMambaBlock):
    def __init__(
        self,
        opt,
        InChannels: int,
        PatchSize: Optional[int]=2,
        NumMambaBlocks: Optional[int]=2,
        DropRate: Optional[float]=0.0,
        Dilation: Optional[int]=1,
        FuseDropRate: Optional[float]=0., 
        UseMamba=True,
        MambaSELayer: Optional[nn.Module]=None,
        **kwargs,
    ) -> None:
        DimEmb = InChannels // 2
        DimCNNOut = DimEmb

        Conv3x3In = Convolution(
            opt, InChannels, InChannels, 3, 1, 
            Groups=InChannels, Dilation=Dilation, ActLayer=nn.SiLU,
        ) # depth-wise separable convolution
        MambaSELayer = MambaSELayer(opt, InChannels, **kwargs) if MambaSELayer is not None else nn.Identity()
        Conv1x1In = Convolution(opt, InChannels, DimCNNOut, 1, 1, UseNorm=False)
        
        super(MobileMambaBlockv1, self).__init__(
            opt, UseMamba, DimEmb, PatchSize, NumMambaBlocks, DropRate, FuseDropRate,
        )
        self.LocalRep = nn.Sequential(Conv3x3In, MambaSELayer, Conv1x1In)

        self.ConvProj = Convolution(opt, DimCNNOut, InChannels, 1, 1)

        self.DimCNNOut = DimCNNOut
        
    def forward(self, x: Tensor) -> Tensor:
        # [B x C x H x W] --> [B x C // 2 x H x W]
        FeatureMap = self.LocalRep(x)

        # global associations
        FeatureMap = self.mambaForward(FeatureMap)
            
        # [B x C // 2 x H x W] --> [B x C x H x W]
        FeatureMap = self.ConvProj(FeatureMap)
        
        return x + self.Dropout(FeatureMap)


class MobileMambaBlockv2(MobileMambaBlockv1):
    def __init__(
        self,
        opt,
        InChannels: int,
        PatchSize: Optional[int]=2,
        NumMambaBlocks: Optional[int]=2,
        DropRate: Optional[float]=0.0,
        Dilation: Optional[int]=1,
        FuseDropRate: Optional[float]=0., 
        UseMamba=True,
        MambaSELayer: Optional[nn.Module]=None,
        **kwargs,
    ) -> None:
        super().__init__(
            opt, InChannels, PatchSize, NumMambaBlocks, DropRate, Dilation,
            FuseDropRate, UseMamba, MambaSELayer, **kwargs,
        )
        # MobileMambav2: input changed from just global to local + global
        self.ConvProj = Convolution(opt, 2 * self.DimCNNOut, InChannels, 1, 1)
        
    def forward(self, x: Tensor) -> Tensor:
        # [B x C x H x W] --> [B x C // 2 x H x W]
        FmConv = self.LocalRep(x)

        # global associations
        FeatureMap = self.mambaForward(FmConv)

        # MobileMambav2: local + global instead of only global
        # [B x C // 2 x H x W] --> [B x C x H x W]
        FeatureMap = self.ConvProj(torch.cat((FeatureMap, FmConv), dim=1))

        return x + self.Dropout(FeatureMap)
    

def getMMambaConfiguration(Mode, UseMamba, BlockVer, **kwargs) -> Dict:
    Config = dict()
    MambaBlockPool = [MobileMambaBlockv2]
    MambaBlock = MambaBlockPool[int(re.findall(r'\d+', BlockVer)[0]) - 1]
    Config["mamba_block"] = partial(MambaBlock, UseMamba=UseMamba)
    
    Mode = Mode.lower()
    if Mode == "xx_small":
        exp_mult = 2
        ConfigLayer = {
            "layer1": {
                "mamba_channels": 32,
                "mamba_blocks": 3,
                "patch_h": 2,  # 8,
                "patch_w": 2,  # 8,
                "stride": 1,
                "mm_expand_ratio": exp_mult,
                "block_type": "vim",
            },
            "layer2": {
                "out_channels": 24,
                "mamba_channels": 48,
                "mamba_blocks": 2,
                "patch_h": 2,  # 8,
                "patch_w": 2,  # 8,
                "stride": 2,
                "mm_expand_ratio": exp_mult,
                "block_type": "vim",
            },
            "layer3": {  # 28x28
                "out_channels": 48,
                "mamba_channels": 64,
                "mamba_blocks": 4,
                "patch_h": 2,  # 8,
                "patch_w": 2,  # 8,
                "stride": 2,
                "mm_expand_ratio": exp_mult,
                "block_type": "vim",
            },
            "layer4": {  # 14x14
                "out_channels": 64,
                "mamba_channels": 80,
                "mamba_blocks": 3,
                "patch_h": 2,  # 4,
                "patch_w": 2,  # 4,
                "stride": 2,
                "mm_expand_ratio": exp_mult,
                "block_type": "vim",
            },
            "layer5": {  # 7x7
                "out_channels": 80,
                "mamba_channels": 96,
                "mamba_blocks": 2,
                "patch_h": 2,
                "patch_w": 2,
                "stride": 2,
                "mm_expand_ratio": exp_mult,
                "block_type": "vim",
            },
            "middle_layer_exp_factor": 1,
            "last_layer_exp_factor": 4
        }
        
    elif Mode == "x_small":
        exp_mult = 4
        ConfigLayer = {
            "layer1": {
                "mamba_channels": 48,
                "mamba_blocks": 3,
                "patch_h": 2,  # 8,
                "patch_w": 2,  # 8,
                "stride": 1,
                "mm_expand_ratio": exp_mult,
                "block_type": "vim",
            },
            "layer2": {
                "out_channels": 48,
                "mamba_channels": 72,
                "mamba_blocks": 2,
                "patch_h": 2,  # 8,
                "patch_w": 2,  # 8,
                "stride": 2,
                "mm_expand_ratio": exp_mult,
                "block_type": "vim",
            },
            "layer3": {  # 28x28
                "out_channels": 64,
                "mamba_channels": 96,
                "mamba_blocks": 4,
                "patch_h": 2,
                "patch_w": 2,
                "stride": 2,
                "mm_expand_ratio": exp_mult,
                "block_type": "vim",
            },
            "layer4": {  # 14x14
                "out_channels": 80,
                "mamba_channels": 120,
                "mamba_blocks": 3,
                "patch_h": 2,
                "patch_w": 2,
                "stride": 2,
                "mm_expand_ratio": exp_mult,
                "block_type": "vim",
            },
            "layer5": {  # 7x7
                "out_channels": 96,
                "mamba_channels": 144,
                "mamba_blocks": 2,
                "patch_h": 2,
                "patch_w": 2,
                "stride": 2,
                "mm_expand_ratio": exp_mult,
                "block_type": "vim",
            },
            "middle_layer_exp_factor": 1,
            "last_layer_exp_factor": 4
        }
        
    elif Mode == "small":
        exp_mult = 4
        ConfigLayer = {
            "layer1": {
                "mamba_channels": 48,
                "mamba_blocks": 3,
                "patch_h": 2,  # 8,
                "patch_w": 2,  # 8,
                "stride": 1,
                "mm_expand_ratio": exp_mult,
                "block_type": "vim", 
            },
            "layer2": {
                "out_channels": 64,
                "mamba_channels": 96,
                "mamba_blocks": 2,
                "patch_h": 2,  # 8,
                "patch_w": 2,  # 8,
                "stride": 2,
                "mm_expand_ratio": exp_mult,
                "block_type": "vim",
            },
            "layer3": {  # 28x28
                "out_channels": 96,
                "mamba_channels": 144,
                "mamba_blocks": 4,
                "patch_h": 2,
                "patch_w": 2,
                "stride": 2,
                "mm_expand_ratio": exp_mult,
                "block_type": "vim",
            },
            "layer4": {  # 14x14
                "out_channels": 128,
                "mamba_channels": 192,
                "mamba_blocks": 3,
                "patch_h": 2,
                "patch_w": 2,
                "stride": 2,
                "mm_expand_ratio": exp_mult,
                "block_type": "vim",
            },
            "layer5": {  # 7x7
                "out_channels": 160,
                "mamba_channels": 240,
                "mamba_blocks": 2,
                "patch_h": 2,
                "patch_w": 2,
                "stride": 2,
                "mm_expand_ratio": exp_mult,
                "block_type": "vim",
            },
            "middle_layer_exp_factor": 1,
            "last_layer_exp_factor": 4
        }
        
    elif Mode == "medium":
        exp_mult = 4
        ConfigLayer = {
            "layer1": {
                "mamba_channels": 96,
                "mamba_blocks": 3,
                "patch_h": 2,  # 8,
                "patch_w": 2,  # 8,
                "stride": 1,
                "mm_expand_ratio": exp_mult,
                "block_type": "vim",
            },
            "layer2": {
                "out_channels": 96,
                "mamba_channels": 144,
                "mamba_blocks": 2,
                "patch_h": 2,  # 8,
                "patch_w": 2,  # 8,
                "stride": 2,
                "mm_expand_ratio": exp_mult,
                "block_type": "vim",
            },
            "layer3": {  # 28x28
                "out_channels": 128,
                "mamba_channels": 192,
                "mamba_blocks": 4,
                "patch_h": 2,
                "patch_w": 2,
                "stride": 2,
                "mm_expand_ratio": exp_mult,
                "block_type": "vim",
            },
            "layer4": {  # 14x14
                "out_channels": 160,
                "mamba_channels": 240,
                "mamba_blocks": 3,
                "patch_h": 2,
                "patch_w": 2,
                "stride": 2,
                "mm_expand_ratio": exp_mult,
                "block_type": "vim",
            },
            "layer5": {  # 7x7
                "out_channels": 192,
                "mamba_channels": 288,
                "mamba_blocks": 2,
                "patch_h": 2,
                "patch_w": 2,
                "stride": 2,
                "mm_expand_ratio": exp_mult,
                "block_type": "vim",
            },
            "middle_layer_exp_factor": 1,
            "last_layer_exp_factor": 4
        }

    else:
        raise NotImplementedError
    
    Config.update(ConfigLayer)
    return Config
