from functools import partial

import torch
from torch import nn, Tensor
from torchvision.ops import stochastic_depth

from .resLinkModules import UpLink
from ....layers import (
    getChannelsbyStage, initActLayer, Convolution, StochasticDepth,
)
from .....utils import makeDivisible, setMethod, callMethod


class FGBottleneck(nn.Module):
    def __init__(
        self,
        opt,
        InChannels: int,
        HidChannels: int=None,
        Expansion: float=2.,
        Stride: int=1,
        Dilation: int=1,
        DropRate: float=0.0, # 0 would be better
        SELayer: nn.Module=None,
        ActLayer: nn.Module=None,
        ViTBlock: nn.Module=None,
        **kwargs,
    ) -> None:
        super().__init__()
        # Feature guide bottleneck
        if HidChannels is None:
            HidChannels = makeDivisible(InChannels * Expansion, 8)
        self.Bottleneck = nn.Sequential(
            Convolution(
                opt, InChannels, HidChannels, 1, ActLayer=nn.ReLU,
            ),
            Convolution(
                opt, HidChannels, HidChannels, 3, Stride, Dilation=Dilation, ActLayer=nn.ReLU,
            ),
            SELayer(
                opt, InChannels=HidChannels, **kwargs,
            ) if SELayer is not None else nn.Identity(),
            Convolution(opt, HidChannels, InChannels, 1)
        )
        
        self.ActLayer = initActLayer(ActLayer) if ActLayer is not None else nn.Identity()
        self.Dropout = StochasticDepth(DropRate) if DropRate > 0 else nn.Identity()
        
        self.ViTLayer = ViTBlock(opt, InChannels, **kwargs) if ViTBlock is not None else nn.Identity()
        
    def forward(self, x: Tensor) -> Tensor:
        Out = self.Bottleneck(x)
        Out = self.ActLayer(x + self.Dropout(Out))
        return self.ViTLayer(Out)


class FGADBottleneck(FGBottleneck):
    def __init__(
        self,
        opt,
        InChannels: int,
        HidChannels: int=None,
        Expansion: float=0.4,
        Stride: int=1,
        Dilation: int=1,
        DropRate: float=0.0,
        SELayer: nn.Module=None,
        ActLayer: nn.Module=None,
        ViTBlock: nn.Module=None,
        **kwargs,
    ) -> None:
        super().__init__(
            opt, InChannels, HidChannels, Expansion, Stride, Dilation,
            DropRate, SELayer, ActLayer, ViTBlock, **kwargs
        )
        # 1% lower performance, faster


class BasicBlock(nn.Module):
    def __init__(
        self,
        opt,
        InChannels: int,
        OutChannels: int,
        Stride: int=1,
        **kwargs,
    ) -> None:
        super().__init__()
        # lower performance with vit
        self.Conv = nn.Sequential(
            Convolution(
                opt, InChannels, OutChannels, 3, Stride, ActLayer=nn.ReLU,
            ),
            Convolution(
                opt, OutChannels, OutChannels, 1, ActLayer=nn.ReLU,
            ),
        )
        
    def forward(self, x: Tensor) -> Tensor:
        return self.Conv(x)


class CSLayer(nn.Module):
    """ cross-scale layer
    lower performance with vit or dropout random attnmask
    """
    def __init__(
        self,
        opt,
        InChannels: int,
        OutChannels: int,
        Stride: int=1,
        NumBlocks: int=1,
        **kwargs,
    ) -> None:
        super().__init__()
        self.NumBlocks = NumBlocks

        Block = BasicBlock
        
        for i in range(NumBlocks + 1):
            if i < NumBlocks:
                OutChannelsC1 = InChannels if i < NumBlocks - 1 else OutChannels
                Layer = Block(
                    opt, InChannels, OutChannelsC1, Stride, **kwargs,
                )
                Name = "Conv%d" % (i + 1)
                setMethod(self, Name, Layer)
            
    def forward(self, x: Tensor) -> Tensor:
        SubFeatureTuple = [x]
        for i in range(self.NumBlocks):
            Name = "Conv%d" % (i + 1)
            Output = callMethod(self, Name)(SubFeatureTuple[-1])
            SubFeatureTuple.append(Output)
        
        return SubFeatureTuple[-1]


class FGLink(nn.Module):
    """ similar to reslink, used in feature guide """
    def __init__(
        self, 
        opt, 
        ModelConfigDict, 
        MaxStage: int=1,
        NumBranches: int=1,
        DropRate: float=0,  
        ViTBlock: nn.Module=None, 
        **kwargs,
    ) -> None:
        super().__init__()
        self.opt = opt
        self.ViTBlock = ViTBlock
        self.DropRate = DropRate
        self.NumBranches = NumBranches
        
        self.UpLink = []
        Expansion = 0.25 if self.opt.seg_feature_guide != 3 else opt.link_expansion
        # self.StageList = [] # different from reslink, it use encoded feature tuple
                
        InChannels = getChannelsbyStage(ModelConfigDict, MaxStage)
        for i, s in enumerate(reversed(range(opt.fg_start_stage, MaxStage))):
            OutChannels = getChannelsbyStage(ModelConfigDict, s)

            UpConv = UpLink(opt, InChannels, OutChannels, Expansion=Expansion)
            setMethod(self, "UpLink%d" % (i + 1), UpConv)
            
            if opt.fg_link == 2:
                CatLinkConv = BasicBlock(opt, OutChannels * 2, OutChannels)
                setMethod(self, "CatLinkConv%d" % (i + 1), CatLinkConv)
                
            ViTLayer = ViTBlock(opt, OutChannels, **kwargs) if ViTBlock is not None else nn.Identity()
            setMethod(self, "ViTLayer%d" % (i + 1), ViTLayer)
            
            InChannels = OutChannels
            # self.StageList.append(s) 
        
        self.stochasticDepth = partial(stochastic_depth, mode="row")
        self.ActLayer = nn.ReLU(inplace=True)

    def forwardFeature(self, FeaturesTuple: list) -> list:
        DecodeFeatureTuple = []
        DecodeFeature = FeaturesTuple[-1]
        for i, f in enumerate(reversed(FeaturesTuple[self.opt.fg_start_stage - 1:-1])):
            UpLink = callMethod(self, "UpLink%d" % (i + 1))
            DecodeFeature = UpLink(DecodeFeature)
            if self.opt.fg_link == 1:
                DecodeFeature = DecodeFeature + f
            else:
                DecodeFeature = torch.cat([DecodeFeature, f], axis=1)
                
                CatLinkConv = callMethod(self, "CatLinkConv%d" % (i + 1))
                DecodeFeature = CatLinkConv(DecodeFeature)
            
            if self.ViTBlock is not None:
                ViTLayer = callMethod(self, "ViTLayer%d" % (i + 1))
                DecodeFeature = ViTLayer(DecodeFeature)
            
            if self.opt.seg_feature_guide == 3:
                BlockID = len(FeaturesTuple) - self.opt.fg_start_stage - i
                DropRate = self.DropRate * float(BlockID) / self.NumBranches
                DecodeFeature = self.ActLayer(
                    f + self.stochasticDepth(
                        DecodeFeature, p=DropRate, training=self.training,
                ))
            
            DecodeFeatureTuple.append(DecodeFeature)
        return DecodeFeatureTuple
        
    def forward(self, FeaturesTuple: list) -> Tensor:
        return self.forwardFeature(FeaturesTuple)[-1]
