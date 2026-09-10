import warnings

from functools import partial
from typing import Union, Tuple

import torch
from torch import nn, Tensor
import torch.nn.functional as F

from .utils import getAuxOut
from ..utils import TaskAbbrevs


class BaseCriteria(nn.Module):
    def __init__(self, opt, UseBackground=True, **kwargs):
        super(BaseCriteria, self).__init__()
        self.opt = opt
        
        self.IgnoreIdx = opt.ignore_idx
        self.UseClsWts = opt.class_weights
        self.Reduction = opt.loss_reduction
        self.SpatialDims = opt.spatial_dims
        self.LabelSmoothing = opt.label_smoothing
        self.NumClasses = getattr(
            opt, "%s_num_classes" % TaskAbbrevs.get(opt.task, opt.num_classes)
        )

        self.eps = 1e-7
        self.UseBackground = UseBackground
        
        self.getAuxOut = getAuxOut

    def oneHot(self, Prediction: Tensor, Target: Tensor) -> Tensor:
        NumClasses = Prediction.shape[1]
        if Prediction.shape[1] > 1:
            # ignore single channel prediction.
            Target = F.one_hot(Target, num_classes=NumClasses)
        return Target
        
    def skipBackground(
        self, Prediction: Tensor, Target: Tensor, AuxOut: Tensor=None
    ) -> Union[Union[Tensor, Tensor], Union[Tensor, Tensor, Tensor]]:
        # for segmentation and detection
        if not self.UseBackground:
            if Prediction.shape[1] == 1:
                warnings.warn("single channel prediction, `UseBackground=False` ignored.")
            else:
                # if skipping background, removing first channel
                Prediction2 = Prediction[:, 1:]
                Target2 = Target[:, 1:] if Target.ndim == 4 else Target
                if AuxOut is not None: 
                    AuxOut = AuxOut[:, 1:]
                    Prediction2 = (Prediction2, AuxOut)
        else:
            Prediction2, Target2 = Prediction, Target
        return Prediction2, Target2

    def forward(
        self, Input: Tensor, Prediction: Tensor, Target: Tensor, **kwargs,
    ) -> Tensor:
        pass

    def classWeights(self, Target: Tensor, NumClasses: int, NormVal: float = 1.1) -> Tensor:
        """ Implementation of a class-weighting scheme,
        of `ENet <https://arxiv.org/pdf/1606.02147.pdf>`_ paper.
        """
        ClassHist = torch.histc(Target.float(), bins=NumClasses, min=0, max=NumClasses - 1)
        MaskIndices = ClassHist == 0

        # normalize between 0 and 1 by dividing by the sum
        NormHist = torch.div(ClassHist, ClassHist.sum())
        NormHist = torch.add(NormHist, NormVal)

        # compute class weights.
        # samples with more frequency will have less weight and vice-versa
        ClassWts = torch.div(torch.ones_like(ClassHist), torch.log(NormHist))

        # mask the classes which do not have samples in the current batch
        ClassWts[MaskIndices] = 0.0

        return ClassWts.to(Target.device)

    def weightForward(self, Target: Tensor):
        # manual rescaling weight for each class, passed to binary Cross-Entropy loss
        return self.classWeights(Target, self.NumClasses) \
            if self.UseClsWts and self.training else None
    
    def lossRedManager(self, Loss: Tensor):
        if self.Reduction == "mean":
            Loss = Loss.mean()
        elif self.Reduction == "sum":
            Loss = Loss.sum()  
        return Loss

    def __repr__(self):
        return "{}()".format(self.__class__.__name__)


class BaseSegCriteria(BaseCriteria):
    def __init__(self, opt, **kwargs):
        super(BaseSegCriteria, self).__init__(opt, **kwargs)
        self.AuxWt = opt.aux_weight
        
        self.ForwardWithWeight = False
        self.InterpolateMode = "trilinear" if opt.spatial_dims == 3 else "bilinear"

    def computeLoss(
        self, Input: Tensor, Mask: Tensor, Target: Tensor, **kwargs,
    ) -> Tensor:
        pass
        
    def forward(
        self, Input: Tensor, Prediction: Union[Tensor, Tuple[Tensor, Tensor]], Target: Tensor, **kwargs,
    ) -> Tensor:
        Mask, AuxOut = self.getAuxOut(Prediction)
        Mask, Target = self.skipBackground(Mask, Target, AuxOut=AuxOut)
        if isinstance(Mask, tuple): Mask, AuxOut = Mask
        
        if self.ForwardWithWeight:
            Weight = self.weightForward(Target)
            self.computeLoss = partial(self.computeLoss, Weight=Weight)
        
        TotalLoss = self.computeLoss(Input=Input, Mask=Mask, Target=Target, **kwargs)
        if self.training and AuxOut is not None:
            LossAux = self.computeLoss(Input=Input, Mask=AuxOut, Target=Target, **kwargs)
            TotalLoss = TotalLoss + (self.AuxWt * LossAux)
            
        return TotalLoss