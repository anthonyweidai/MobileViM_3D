from torch import nn, Tensor

from ....layers import Convolution, TransposeConvLayer
from .....utils import makeDivisible


class UpLink(nn.Module):
    def __init__(
        self, opt,
        InChannels, OutChannels, HidChannels: int=None, Expansion: float=0.25,
    ):
        super(UpLink, self).__init__()
        if HidChannels is None:
            HidChannels = makeDivisible(InChannels * Expansion, 8)
        
        self.TPConv = nn.Sequential(
            Convolution(
                opt, InChannels, HidChannels, 1, 1, 0, ActLayer=nn.ReLU,
            ),
            TransposeConvLayer(
                opt, HidChannels, HidChannels, 3, 2, 1, 1, ActLayer=nn.ReLU,
            ),
            Convolution(
                opt, HidChannels, OutChannels, 1, 1, 0, ActLayer=nn.ReLU,
            )
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.TPConv(x)
