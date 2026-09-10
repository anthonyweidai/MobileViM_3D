from torch import nn
from torchvision.ops import StochasticDepth as StochasticDepthTorch


class StochasticDepth(StochasticDepthTorch):
    def __init__(self, p: float, Mode: str="row") -> None:
        super().__init__(p, Mode)
        # drop out layer
        
    
class Dropout(nn.Dropout):
    def __init__(self, p: float=0.5, inplace: bool=False):
        super(Dropout, self).__init__(p=p, inplace=inplace)
        # randomly zeroes some of the elements of the input tensor