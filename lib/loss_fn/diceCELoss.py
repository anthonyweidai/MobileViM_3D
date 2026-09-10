from typing import Tuple, Union

import torch
from torch import Tensor
from torch.nn import functional as F

from .baseCriteria import BaseSegCriteria
from .lossFnRegister import LOSS_FN_REGISTRY


@LOSS_FN_REGISTRY.register("cross_entropy", "segmentation")
class SegCrossEntropy(BaseSegCriteria):
    """Cross entropy loss for the task of semantic segmentation"""

    def __init__(self, opt, **kwargs):
        super(SegCrossEntropy, self).__init__(opt, **kwargs)
        self.ForwardWithWeight = True

    def computeLoss(self, Mask: Tensor, Target: Tensor, Weight=None, **kwargs):
        if Target.ndim == (self.SpatialDims + 2):
            Target2 = torch.argmax(Target, dim=1)
        else:
            Target2 = Target
            
        # use label smoothing only for training
        LabelSmoothing = self.LabelSmoothing if self.training else 0.0
        
        if Mask.shape[-self.SpatialDims:] != Target2.shape[-self.SpatialDims:]:
            Prediction2 = F.interpolate(
                Mask, size=Target2.shape[-self.SpatialDims:], 
                mode=self.InterpolateMode, align_corners=True
            )
        else:
            Prediction2 = Mask

        return F.cross_entropy(
            input=Prediction2,
            target=Target2,
            weight=Weight,
            ignore_index=self.IgnoreIdx,
            reduction=self.Reduction,
            label_smoothing=LabelSmoothing,
        )


@LOSS_FN_REGISTRY.register("nnunet_dice", "segmentation")
class SoftDiceLoss(BaseSegCriteria):
    def __init__(
        self, opt, batch_dice: bool=False, do_bg: bool=True, 
        smooth: float=1., clip_tp: float=None, **kwargs,
    ):
        """ Adapted from
        https://github.com/MIC-DKFZ/nnUNet/blob/19ae215bc8fef6a95fdc166f9fc305558f2e3a74/nnunetv2/training/loss/dice.py
        """
        super(SoftDiceLoss, self).__init__(opt, **kwargs)
        
        self.do_bg = do_bg
        self.batch_dice = batch_dice
        self.smooth = smooth
        self.clip_tp = clip_tp
    
    def apply_nonlin(self, x: Tensor) -> Tensor:
        return torch.softmax(x, 1)

    def get_tp_fp_fn_tn(self, net_output, gt, axes=None, mask=None, square=False):
        """
        net_output must be (b, c, x, y(, z)))
        gt must be a label map (shape (b, 1, x, y(, z)) OR shape (b, x, y(, z))) or one hot encoding (b, c, x, y(, z))
        if mask is provided it must have shape (b, 1, x, y(, z)))
        :param net_output:
        :param gt:
        :param axes: can be (, ) = no summation
        :param mask: mask must be 1 for valid pixels and 0 for invalid pixels
        :param square: if True then fp, tp and fn will be squared before summation
        :return:
        """
        if axes is None:
            axes = tuple(range(2, net_output.ndim))

        with torch.no_grad():
            if net_output.ndim != gt.ndim:
                gt = gt.view((gt.shape[0], 1, *gt.shape[1:]))

            if net_output.shape == gt.shape:
                # if this is the case then gt is probably already a one hot encoding
                y_onehot = gt
            else:
                y_onehot = torch.zeros(net_output.shape, device=net_output.device, dtype=torch.bool)
                y_onehot.scatter_(1, gt.long(), 1)

        tp = net_output * y_onehot
        fp = net_output * (~y_onehot)
        fn = (1 - net_output) * y_onehot
        tn = (1 - net_output) * (~y_onehot)

        if mask is not None:
            with torch.no_grad():
                if mask.ndim == (self.SpatialDims + 1):
                    mask_here = mask.unsqueeze(1)
                else:
                    mask_here = torch.tile(mask, (1, tp.shape[1], *[1 for _ in range(2, tp.ndim)]))
            tp *= mask_here
            fp *= mask_here
            fn *= mask_here
            tn *= mask_here
            
        if square:
            tp = tp ** 2
            fp = fp ** 2
            fn = fn ** 2
            tn = tn ** 2

        if len(axes) > 0:
            tp = tp.sum(dim=axes, keepdim=False)
            fp = fp.sum(dim=axes, keepdim=False)
            fn = fn.sum(dim=axes, keepdim=False)
            tn = tn.sum(dim=axes, keepdim=False)

        return tp, fp, fn, tn
        
    def computeLoss(
        self, Mask: Tensor, Target: Tensor, loss_mask=None, **kwargs
    ) -> Tensor:
        shp_x = Mask.shape
        
        if self.batch_dice:
            axes = [0] + list(range(2, len(shp_x)))
        else:
            axes = list(range(2, len(shp_x)))

        Prediction2 = self.apply_nonlin(Mask) if self.apply_nonlin is not None else Mask
            
        tp, fp, fn, _ = self.get_tp_fp_fn_tn(Prediction2, Target, axes, loss_mask, False)

        if self.clip_tp is not None:
            tp = torch.clip(tp, min=self.clip_tp , max=None)

        nominator = 2 * tp
        denominator = 2 * tp + fp + fn

        dc = (nominator + self.smooth) / (torch.clip(denominator + self.smooth, 1e-8))

        if not self.do_bg:
            if self.batch_dice:
                dc = dc[1:]
            else:
                dc = dc[:, 1:]
        dc = dc.mean()
        
        return - dc
    

@LOSS_FN_REGISTRY.register("dice_ce", "segmentation")
class CDWithCE(BaseSegCriteria):
    " Dice + cross-entropy loss "
    def __init__(self, opt, **kwargs):
        super(CDWithCE, self).__init__(opt, **kwargs)
        self.WeightCE = 1
        self.WeightDice = 1
        
        self.CrossEntropyLoss = SegCrossEntropy(opt, **kwargs)
        self.DiceLoss = SoftDiceLoss(opt, **kwargs)

    def computeLoss(
        self, Input: Tensor, Mask: Tensor, Target: Tensor, 
        TargetDice: Tensor, LossMask: Tensor, Weight=None,
    ) -> Tensor:
        return self.WeightCE * self.CrossEntropyLoss(Input, Mask, Target, Weight=Weight) + \
            self.WeightDice * self.DiceLoss(Input, Mask, TargetDice, loss_mask=LossMask)
    
    def forward(
        self, Input: Tensor, Prediction: Union[Tensor, Tuple[Tensor, Tensor]], Target: Tensor, **kwargs,
    ) -> Tensor:
        Prediction2, AuxOut = self.getAuxOut(Prediction)
        Prediction2, Target2 = self.skipBackground(Prediction2, Target, AuxOut=AuxOut)
        if isinstance(Prediction2, tuple): Prediction2, AuxOut = Prediction2
        
        if self.IgnoreIdx >= 0:
            # remove ignore label from Target, replace with one of the known labels. 
            # It doesn't matter because we ignore gradients in those areas anyway
            LossMask = Target2 != self.IgnoreIdx
            TargetDice = torch.where(LossMask, Target2, 0)
        else:
            LossMask = None
            TargetDice = Target2
        
        Weight = self.weightForward(Target2)
        TotalLoss = self.computeLoss(Input, Prediction2, Target2, TargetDice, LossMask, Weight=Weight)
        if self.training and AuxOut is not None:
            LossAux = self.computeLoss(Input, AuxOut, Target2, TargetDice, LossMask, Weight=Weight)
            TotalLoss = TotalLoss + (self.AuxWt * LossAux)
            
        return TotalLoss