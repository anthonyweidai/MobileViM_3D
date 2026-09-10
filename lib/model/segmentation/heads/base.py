from typing import Tuple, Union

from torch import nn, Tensor
from torch.nn import functional as F

from .utils import HEAD_OUT_CHANNELS
from ...layers import (
    checkExp, computeMinStage, computeMaxStage, getLastIdxFromStage, getAllLayerIndex, 
    getChannelsbyLayer, getChannelsbyStage, getAllStageOut, getAllOutChannels,
)
from ....utils import pair


class BaseModule(nn.Module):
    def __init__(self, opt, ModelConfigDict=None, **kwargs):
        super().__init__()
        # separate base modules and fg modules
        # opt.fg_start_stage = max(opt.fg_start_stage, computeMinStage(ModelConfigDict))
        
        self.opt = opt
        self.SpatialDims = opt.spatial_dims
        self.NumClasses = opt.seg_num_classes
        self.ModelConfigDict = ModelConfigDict
        self.InterpolateMode = "bilinear" if self.SpatialDims == 2 else "trilinear"
        
        self.checkExp = checkExp
        self.getAllStageOut = getAllStageOut
        self.computeMinStage = computeMinStage
        self.computeMaxStage = computeMaxStage
        self.getAllLayerIndex = getAllLayerIndex
        self.getAllOutChannels = getAllOutChannels
        self.getChannelsbyLayer = getChannelsbyLayer
        self.getChannelsbyStage = getChannelsbyStage
        self.getLastIdxFromStage = getLastIdxFromStage
        
        self.Classifier = None
        # default out channels
        self.OutChannels = HEAD_OUT_CHANNELS.get(opt.seg_head_name, HEAD_OUT_CHANNELS["default"])
    
    def forwardDecode(self, FeaturesTuple: list[Tensor]) -> Tensor:
        pass

    def forwardSegHead(self, FeaturesTuple: list) -> Tensor: 
        Output = self.forwardDecode(FeaturesTuple)
        Output = self.Classifier(Output)
        return F.interpolate(
            Output, size=pair(self.opt.aug_shape, RepNum=self.SpatialDims), 
            mode=self.InterpolateMode, align_corners=True,
        )

    def forward(
        self, FeaturesTuple, **kwargs
    ) -> Union[Tensor, Tuple[Tensor], dict[Tensor]]:
        if self.opt.seg_feature_guide == 0:
            Output = self.forwardSegHead(FeaturesTuple)
        else:
            Output = self.forwardFMG(FeaturesTuple)
            
        return Output
    
    def forwardFMG(self, FeaturesTuple: list[Tensor]) -> Union[Tensor, Tuple[Tensor]]:
        pass