from typing import Callable, Optional, List, Dict

from torch import nn, Tensor

from .modules import getMMambaConfiguration
from .. import MODEL_REGISTRY
from ..modules import BaseEncoder
from ..layers import (
    initWeight, Convolution, Linearlayer,
    Globalpooling, Dropout, StochasticDepth,
)
from ...utils import setMethod, makeDivisible


class MambaSEConv(nn.Module):
    def __init__(
        self, 
        opt,
        InChannels: int, 
        OutChannels: int, 
        ExpRatio: float, 
        Kernel: int, 
        Stride: int,
        DropRate: float=0., 
        # WidthMult: float=1.0, 
        SELayer: Callable[..., nn.Module]=None,
        SEActLayer: Callable[..., nn.Module]=nn.SiLU,
        **kwargs
    ) -> None:
        super().__init__()
        Layers = nn.ModuleList()
        
        # expand
        ExpChannels = makeDivisible(InChannels * ExpRatio, 8)
        if ExpChannels != InChannels:
            Layers.append(Convolution(opt, InChannels, ExpChannels, 1, ActLayer=SEActLayer))

        # depthwise
        Layers.append(Convolution(
            opt, ExpChannels, ExpChannels, Kernel, Stride, 
            Groups=ExpChannels, ActLayer=SEActLayer,
        ))

        # squeeze and excitation
        SqueezeFactor = 4
        HidChannels = makeDivisible(InChannels // SqueezeFactor, 8)
        Layers.append(
            SELayer(
                opt=opt, InChannels=ExpChannels, HidChannels=HidChannels, 
                Act=SEActLayer, **kwargs,
            ) if SELayer else nn.Identity()
        )
        
        # project
        Layers.append(Convolution(opt, ExpChannels, OutChannels, 1,))
        self.Block = nn.Sequential(*Layers)
        
        self.ResConnect = Stride == 1 and InChannels == OutChannels
        if self.ResConnect: self.Dropout = StochasticDepth(DropRate)

    def forward(self, x: Tensor) -> Tensor:
        Result = self.Block(x)
        
        if self.ResConnect:
            Result = x + self.Dropout(Result)
        return Result


class MobileMamba(BaseEncoder):
    def __init__(
        self,
        opt,
        MobileMambaConfig: dict,
        EndExp: Optional[bool]=True,
        **kwargs
    ):
        OutChannels = 16
        super(MobileMamba, self).__init__(opt, **kwargs)
        
        self.BlockID = 0
        self.TotalBlocks = 6
        self.StridedStage = 1
        self.FirstOuChannel = OutChannels
        self.MambaBlock = MobileMambaConfig["mamba_block"]
        
        self.SELayer = None
        
        self.ModelConfigDict = dict()
        self.Conv1 = Convolution(
            opt, opt.in_channels, OutChannels, 3, 2, ActLayer=nn.SiLU,
        )
        self.ModelConfigDict["conv1"] = {"in": 3, "out": OutChannels, "stage": 1}
        
        for i in range(5):
            InChannels = OutChannels
            Layer, OutChannels = self.makeLayer(
                InChannels=InChannels, Cfg=MobileMambaConfig["layer%d" % (i + 1)], **kwargs
            )
            self.ModelConfigDict["layer%d" % (i + 1)] = {"in": InChannels, "out": OutChannels, "stage": i + 1}
            setMethod(self, "Layer%d" % (i + 1), Layer)
        
        # Add one learnable layer
        if EndExp and "classification" in opt.task:
            InChannels = OutChannels
            self.ExpChannel = min(MobileMambaConfig["last_layer_exp_factor"] * InChannels, 960)
            self.Exp1x1 = Convolution(
                opt, InChannels, self.ExpChannel, 1, 1, ActLayer=nn.SiLU,
            )
            self.ModelConfigDict["exp1x1"] = {"in": InChannels, "out": self.ExpChannel, "stage": 6}
            
            # Global pool to linear
            self.Classifier = nn.Sequential(
                Globalpooling(PoolType="mean", SpatialDims=opt.spatial_dims),
                Dropout(0.1, inplace=True),
                Linearlayer(self.ExpChannel, opt.num_classes, bias=True),
            )
        
        if opt.init_weight: self.apply(initWeight)
        
    def makeLayer(self, InChannels: int, Cfg: Dict, **kwargs):
        if Cfg["stride"] == 2: self.StridedStage += 1
        
        BlockType = Cfg.get("block_type", "vim")
        if BlockType.lower() == "vim":
            return self.makeLMambaLayer(InChannels, Cfg, **kwargs)
        else:
            return self.makeMobileBlockLayer(InChannels, Cfg, **kwargs)
    
    def makeLMambaLayer(self, InChannels: int, Cfg: Dict, **kwargs):
        Block = []
        Stride = Cfg.get("stride", 1)
        OutChannels = Cfg.get("out_channels")
        ExpandRatio = Cfg.get("mv_expand_ratio", 4)
        
        if Stride == 2:
            # input_channels, out_channels, kernel, stride, expand_ratio
            Layer = MambaSEConv(
                self.opt, InChannels, OutChannels, ExpandRatio, 3, Stride, 
                SELayer=self.SELayer, **kwargs,
            )
            Block.append(Layer)
            InChannels = OutChannels
        
        # drop rate decay
        self.BlockID += 1
        DropRate = self.opt.init_drop_rate1 * self.BlockID / self.TotalBlocks
        FuseDropRate = self.opt.init_drop_rate2 * self.BlockID / self.TotalBlocks
        
        Block.append(
            self.MambaBlock(
                self.opt,
                InChannels,
                DimEmb=Cfg["mamba_channels"],
                PatchSize=(Cfg.get("patch_h", 2), Cfg.get("patch_w", 2)),
                NumMambaBlocks=Cfg["mamba_blocks"],
                ConvKSize=3,
                DropRate=DropRate,
                FuseDropRate=FuseDropRate,
                **kwargs,
            )
        )
        
        return nn.Sequential(*Block), InChannels
    
    def makeMobileBlockLayer(self, InChannels: int, Cfg: Dict, **kwargs):
        OutChannels = Cfg.get("out_channels")
        NumBlocks = Cfg.get("num_blocks", 2)
        ExpandRatio = Cfg.get("expand_ratio", 4)
        Block: List[nn.Module] = []
        
        for i in range(NumBlocks):
            Stride = Cfg.get("stride", 1) if i == 0 else 1 # The first one (64 x 64) is perform down-sampling
            
            if Stride == 1: self.BlockID += 1
            FuseDropRate = self.opt.init_drop_rate2 * self.BlockID / self.TotalBlocks
            # input_channels, out_channels, kernel, stride, expand_ratio
            Layer = MambaSEConv(
                self.opt, InChannels, OutChannels, ExpandRatio, 3, Stride,
                DropRate=FuseDropRate, SELayer=self.SELayer, **kwargs,
            )
            
            Block.append(Layer)
            InChannels = OutChannels
        return nn.Sequential(*Block), InChannels

    def forwardHead(self, x: Tensor) -> Tensor:
        return self.Classifier(x)
    
    
@MODEL_REGISTRY.register("mobilevimxxs", "classification")
def mobieMambaxxs(opt, **kwargs):
    Mode = "xx_small"
    BlockVer = "v%d" % opt.mamba_block
    MobileMambaConfig = getMMambaConfiguration(Mode=Mode, UseMamba=opt.use_mamba, BlockVer=BlockVer)
    return MobileMamba(opt, MobileMambaConfig, **kwargs)


@MODEL_REGISTRY.register("mobilevimxs", "classification")
def mobieMambaxs(opt, **kwargs):
    Mode = "x_small"
    BlockVer = "v%d" % opt.mamba_block
    MobileMambaConfig = getMMambaConfiguration(Mode=Mode, UseMamba=opt.use_mamba, BlockVer=BlockVer)
    return MobileMamba(opt, MobileMambaConfig, **kwargs)


@MODEL_REGISTRY.register("mobilevims", "classification")
def mobieMambas(opt, **kwargs):
    Mode = "small"
    BlockVer = "v%d" % opt.mamba_block
    MobileMambaConfig = getMMambaConfiguration(Mode=Mode, UseMamba=opt.use_mamba, BlockVer=BlockVer)
    return MobileMamba(opt, MobileMambaConfig, **kwargs)


@MODEL_REGISTRY.register("mobilevimm", "classification")
def mobieMambam(opt, **kwargs):
    Mode = "medium"
    BlockVer = "v%d" % opt.mamba_block
    MobileMambaConfig = getMMambaConfiguration(Mode=Mode, UseMamba=opt.use_mamba, BlockVer=BlockVer)
    return MobileMamba(opt, MobileMambaConfig, **kwargs)