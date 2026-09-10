import torch
from torch import Tensor

from .baseNetwork import BaseNetwork
from ..layers import AdaptiveAvgPool
from ...utils import callMethod


class BaseEncoder(BaseNetwork):
    def __init__(self, opt, **kwargs):
        super().__init__(opt, **kwargs)
        self.NumClasses = opt.cls_num_classes
        self.AvgPooling = AdaptiveAvgPool(1, SpatialDims=self.SpatialDims)

    def forwardByConfig(self, x: Tensor) -> list:
        # may not update self layers' weight
        pass
    
    def forwardTuple(self, x: Tensor) -> list[Tensor]:
        # feature tuple occupies the same memory wih pin_memory is ture
        FeaturesTuple = list()
        for k in self.ModelConfigDict:
            Layer = callMethod(self, k.capitalize())
            x = Layer(x)
            FeaturesTuple.append(x)
        return FeaturesTuple
    
    def forwardHead(self, x: Tensor) -> Tensor:
        x = self.AvgPooling(x)
        x = torch.flatten(x, 1)
        return self.Classifier(x)