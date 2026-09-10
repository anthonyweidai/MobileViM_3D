import re
import math

from torch import Tensor

from .baseNetwork import BaseNetwork
from ...utils import pair, callMethod


class BaseMambaEncoder(BaseNetwork):
    def __init__(self, opt, **kwargs):
        super().__init__(opt, **kwargs)
        self.NumClasses = opt.cls_num_classes
        self.PatchSize = pair(int(re.findall(r"patch(\d+)", opt.model_name)[0]))
        self.GridSize = [s // p for s, p in zip(self.ImgShape, self.PatchSize)]
        self.NumPatches = math.prod(self.GridSize)

        self.PatchLen = None
        self.CodePatch = None
        self.InterFactor = None
        
        # vim has only one layer stage
        self.ModelConfigDict["layer1"] = \
            {"in": opt.in_channels, "out": opt.in_channels, "stage": 1}
        
    def featureEmbedding(self, x: Tensor) -> Tensor:
        # pre-processing/positional embedding for input data
        pass
    
    def forwardTuple(self, x: Tensor) -> list[Tensor]:
        # feature tuple occupies the same memory wih pin_memory is ture
        x = self.featureEmbedding(x)
        
        FeaturesTuple = list()
        for k in self.ModelConfigDict:
            Layer = callMethod(self, k.capitalize())
            x = Layer(x)
            FeaturesTuple.append(x)
        return FeaturesTuple