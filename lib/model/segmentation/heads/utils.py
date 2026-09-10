import torch.nn as nn
from torch import Tensor

from ...layers import Convolution


# resize shape 512 x 512 only
HEAD_OUT_CHANNELS = {"default": 512}


class SegHeadClassifier(nn.Module):
    def __init__(self, opt, FMGChannels, ClsInChannels, NumClasses) -> None:
        super().__init__()
        # use expansion will dramatically increase the model size
        if FMGChannels is not None: 
            ClsInChannels += FMGChannels
        
        self.Classifier = Convolution(opt, ClsInChannels, NumClasses, 1, 1, UseNorm=False) # 24.639 M, 24023.218 M
        self.ClsInChannels = ClsInChannels
    
    def forward(self, x: Tensor) -> Tensor:
        return self.Classifier(x)