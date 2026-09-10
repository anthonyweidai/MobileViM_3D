import numpy as np
from typing import Dict, Union

from monai.data.meta_tensor import MetaTensor
from monai.transforms import EnsureChannelFirst
from monai.utils.enums import TransformBackends
from monai.transforms.transform import Transform as MTransform

import torch
from torch import Tensor
from torchvision import transforms as T


""" for the resize of 3D data """
class ChannelFirst(MTransform):
    """ Adjust or add the channel dimension of input data to ensure `channel_first` shape. """
    backend = [TransformBackends.TORCH, TransformBackends.NUMPY]
    def __init__(self, InChannels: int, ChannelDim: int = None, StrictCheck: bool=True) -> None:
        super().__init__()
        self.InChannels = InChannels
        self.ChannelDim = ChannelDim
        self.Adjuster = EnsureChannelFirst(strict_check=StrictCheck, channel_dim=ChannelDim)

    def __call__(self, Img, MetaDict=None):
        """
        Apply the transform to `img`.
        """
        if isinstance(Img, (np.ndarray, Tensor)):
            ImgShape = list(Img.shape)
            if self.InChannels in ImgShape:
                Img = self.Adjuster(Img, meta_dict=MetaDict)
            elif self.InChannels == 1:
                if isinstance(Img, np.ndarray):
                    Img = np.expand_dims(Img, axis=0)
                else:
                    Img.unsqueeze_(0)
        else:
            raise NotImplementedError
            
        return Img


class SqueezeChannel(MTransform):
    """ Squeeze the channel dimension of input data to restore channel shape. """
    backend = [TransformBackends.TORCH, TransformBackends.NUMPY]
    def __init__(self, InChannels: int, ChannelDim: int = None, StrictCheck: bool=True) -> None:
        super().__init__()
        self.InChannels = InChannels
        self.ChannelDim = ChannelDim

    def __call__(self, Img, MetaDict=None):
        """ Apply the transform to `img`. """
        if isinstance(Img, (np.ndarray, Tensor)):
            ImgShape = list(Img.shape)
            if self.InChannels in ImgShape and self.InChannels == 1:
                if isinstance(Img, np.ndarray):
                    Img = np.squeeze(Img, axis=0)
                else:
                    Img.squeeze_(0)
        else:
            raise NotImplementedError
        return Img


class MetaTensor2Tensor(T.ToTensor):
    def __init__(self) -> None:
        super().__init__()
        """ Convert tensor subclass to the parent class
        SyncBatchNorm doesn't work with subclass of torch.Tensor
        https://github.com/pytorch/pytorch/issues/86456
        """
        
    def __call__(self, Data: Union[Dict, MetaTensor]) -> Union[Dict, Tensor]:
        if isinstance(Data, Dict):
            for k, v in Data.items():
                if isinstance(v, MetaTensor):
                    Data[k] = v.as_tensor()
            if "mask" in Data:
                Data["mask"] = Data["mask"].type(dtype=torch.long)
        else:
            Data = Data.as_tensor()
        return Data