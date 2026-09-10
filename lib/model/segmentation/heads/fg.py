from torch import Tensor

from . import SEGHEAD_REGISTRY
from .baseSegHead import BaseSegHead
from .utils import SegHeadClassifier
from ...layers import initWeight


@SEGHEAD_REGISTRY.register("fg")
class FeatureMapGuide(BaseSegHead):
    """ The common segmentation module of feature map guide """
    def __init__(self, opt, **kwargs) -> None:
        super().__init__(opt, **kwargs)
        if self.FMGChannels is None:
            self.FMGChannels = self.getChannelsbyStage(self.ModelConfigDict, -1)
        
        self.Classifier = SegHeadClassifier(
            opt, self.FMGChannels, 0, self.NumClasses, 
        )

        if opt.init_weight: self.apply(initWeight)
        
        self.OutChannels = self.Classifier.ClsInChannels
            
    def forwardDecode(self, FeaturesTuple: list) -> Tensor:
        return FeaturesTuple[-1]
