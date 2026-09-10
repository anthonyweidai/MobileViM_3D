from typing import Optional, Union, Tuple

from torch import nn, Tensor

from ..layers import NormLayerTuple
from ..modules import BaseEncoder, BaseMambaEncoder


class BaseSegmentation(nn.Module):
    """ Base class for segmentation networks """
    def __init__(
        self, opt, 
        Encoder: Optional[Union[BaseEncoder, BaseMambaEncoder]],
        **kwargs,
    ) -> None:
        super(BaseSegmentation, self).__init__()
        assert isinstance(
            Encoder, Union[BaseEncoder, BaseMambaEncoder]
        ), "Encoder should be an instance of BaseEncoder or BaseViTEncoder"
        self.opt = opt
        self.Encoder: Union[BaseEncoder, BaseMambaEncoder] = Encoder
        
        self.SpatialDims = opt.spatial_dims
        self.InterpolateMode = "bilinear" if self.SpatialDims == 2 else "trilinear"
    
    def profileModel(self, Input: Tensor) -> Optional[Tuple[Tensor, float, float]]:
        """
        Child classes must implement this function to compute FLOPs and parameters
        """
        raise NotImplementedError

    def freezeNormLayers(self) -> None:
        for m in self.modules():
            if isinstance(m, NormLayerTuple):
                m.eval()
                m.weight.requires_grad = False
                m.bias.requires_grad = False
                m.training = False
                
    def preprocessInput(self, x: Tensor) -> Tensor:
        if isinstance(x, dict):
            Input = x["image"]
        elif isinstance(x, Tensor):
            Input = x
        else:
            raise NotImplementedError(
                "Input to segmentation should be either a Tensor or a Dict of Tensors"
            )
        return Input