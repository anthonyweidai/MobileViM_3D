from monai.config import KeysCollection
from monai.transforms import EnsureChannelFirst, EnsureChannelFirstd

from ...utils import ChannelFirst, SqueezeChannel


class ChannelFirstd(EnsureChannelFirstd):
    """ Dictionary-based wrapper of channel first """
    backend = EnsureChannelFirst.backend
    def __init__(
        self, Keys: KeysCollection, InChannels: int, ChannelDim=None, 
        StrictCheck: bool=True, AllowMissingKeys: bool=False,
    ) -> None:
        super().__init__(Keys, StrictCheck, AllowMissingKeys, ChannelDim)
        self.adjuster = ChannelFirst(InChannels, ChannelDim, StrictCheck)


class SqueezeChanneld(EnsureChannelFirstd):
    """ Dictionary-based wrapper of channel first """
    backend = EnsureChannelFirst.backend
    def __init__(
        self, Keys: KeysCollection, InChannels: int, ChannelDim=None, 
        StrictCheck: bool=True, AllowMissingKeys: bool=False,
    ) -> None:
        super().__init__(Keys, StrictCheck, AllowMissingKeys, ChannelDim)
        self.adjuster = SqueezeChannel(InChannels, ChannelDim, StrictCheck)