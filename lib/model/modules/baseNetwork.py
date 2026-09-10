from torch import nn, Tensor

from ..layers import buildConfigByOutIds
from ...utils import pair


class BaseNetwork(nn.Module):
    def __init__(self, opt, **kwargs):
        super().__init__()
        self.opt = opt
        
        self.InChannels = opt.in_channels
        self.SpatialDims = opt.spatial_dims
        self.ImgShape = pair(opt.aug_shape, RepNum=opt.spatial_dims)
        
        self.Classifier = None
        
        self.ModelConfigDict = dict()
        """
        Key: conv1, layer*no.
        Values: "in": inchannels, "out": outchannels, "stage": stride stage
        """
        self.buildConfigByOutIds = buildConfigByOutIds

    def checkModel(self):
        assert self.Classifier is not None, "Please implement self.Classifier"
    
    def forwardTuple(self, x: Tensor) -> list[Tensor]:
        # feature tuple occupies the same memory wih pin_memory is ture
        pass
    
    def forwardFeatures(self, x: Tensor) -> Tensor:
        # vit does not change the feature size in this process
        # support deeplabv3, pspnet in segmentaiton
        return self.forwardTuple(x)[-1]

    def forwardHead(self, x: Tensor) -> Tensor:
        pass

    def forward(self, x: Tensor) -> Tensor:
        x = self.forwardFeatures(x)
        return self.forwardHead(x)