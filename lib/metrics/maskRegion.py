import numpy as np
from numbers import Number
from typing import Tuple, Union, Dict

import torch
from torch import Tensor

from .baseMetric import AverageMetric
from .metricRegister import METRIC_REGISTRY


def computeMaskRegion(
    Prediction: Union[dict[Tensor], Tensor], Target: Tensor, 
    SpatialDims: int=2, KeepBackground=True, Eps=1e-7, Mode=True,
):
    """ Manage mask data structure for Jaccard/iou, dice
    Mode for direct calling, tinyObjectsValidation
    """
    Prediction2 = Prediction["mask"] if isinstance(Prediction, Dict) else Prediction
    Target2 = Target["mask"] if isinstance(Target, Dict) else Target
    
    if isinstance(Prediction2, (Tuple, list)) and len(Prediction2) == 2:
        Prediction2 = Prediction2[0]
        assert isinstance(Prediction2, Tensor), "Predicted mask should be a tensor"
        
    # correct dimensions
    NumClasses = Prediction2.shape[1]
    if Prediction2.ndim == (SpatialDims + 2):
        Prediction2 = torch.argmax(Prediction2, dim=1)
    if Target2.ndim == (SpatialDims + 2):
        Target2 = torch.argmax(Target2, dim=1)
    
    # convert to torch.uint8
    PredMask = Prediction2.byte()
    TarMask = Target2.byte()
    # shift by 1 so that 255 is 0
    if KeepBackground:
        PredMask += 1
        TarMask += 1
    # keep the foreground classes
    PredMask = PredMask * (TarMask > 0)

    # calculate mask regions
    Inter = PredMask * (PredMask == TarMask)
    AreaInter = Inter.float().histc(bins=NumClasses, min=1, max=NumClasses)
    # torch.histc(Inter.float(), bins=NumClasses, min=1, max=NumClasses)
    AreaGT = TarMask.float().histc(bins=NumClasses, min=1, max=NumClasses)
    AreaPred = PredMask.float().histc(bins=NumClasses, min=1, max=NumClasses)
    AreaUnion = AreaGT + AreaPred - AreaInter + Eps
    
    # calculate unioncount (semantic level)
    NpTarget = TarMask.cpu().numpy()
    UnionCount = np.zeros((NumClasses, 1), dtype=int)
    
    Areas = np.apply_along_axis(
        lambda a: np.histogram(a, bins=NumClasses, range=(1, NumClasses))[0], 
        1, NpTarget.reshape(*NpTarget.shape[:-SpatialDims], -1)
    )
    UnionIdx = np.where(Areas > Eps, 1, 0)
    UnionCount += np.expand_dims(np.sum(UnionIdx, axis=0), axis=-1)
    
    if Mode:
        return (
            AreaPred.cpu().numpy(), AreaInter.cpu().numpy(), 
            AreaUnion.cpu().numpy(), UnionCount,
        )
    else:
        # create dict with the name (same as)/by varibale name
        return dict(
            count=UnionCount, pred=AreaPred, inter=AreaInter, union=AreaUnion,
        )


@METRIC_REGISTRY.register("mask_region", "segmentation")
class MaskRegion(AverageMetric):
    def gatherMetrics(
        self, Prediction: Tensor, Target: Tensor, **kwargs,
    ) -> Union[Tensor, dict[str, Tensor]]:
        return computeMaskRegion(Prediction, Target, self.SpatialDims, self.Eps, Mode=False)

    def compute(self, **kwargs) -> dict[str, Number]:
        IoU = self.Value["inter"] / self.Value["union"] # summation wihtin an epoch
        Dice = (2 * self.Value["inter"]) / (self.Value["union"] + self.Value["inter"])
        OutputDict = dict(count=self.Value["count"], iou=IoU, dice=Dice)
        
        if not self.IsTraining:
            # AreaUnion - AreaPred + AreaInter = AreaGT
            # FN is true in ground truth: assume negative is false, then it is ture
            Recall = self.Value["inter"] / (self.Value["union"] - self.Value["pred"] + self.Value["inter"] + self.Eps) # TP / (TP + FN)
            Precision = self.Value["inter"] / (self.Value["pred"] + self.Eps) # TP / (TP + FP)
            F1Score = 2. * (Precision * Recall) / (Precision + Recall + self.Eps) # 2TP / (TP + FP)
            F2Score = 5. * (Precision * Recall) / (4. * Precision + Recall + self.Eps) # 5TP / (4TP + FP)
            OutputDict.update(dict(recall=Recall, precision=Precision, f1score=F1Score, f2score=F2Score))
            
            if self.SpatialDims == 3:
                # volume similarity
                # Eq. (21) in https://doi.org/10.1186/s12880-015-0068-x
                VolumeSimilarity = 1 - np.abs(self.Value["union"] - 2. * self.Value["pred"] + self.Value["inter"]) \
                    / (self.Value["union"] + self.Value["inter"] + self.Eps) # 1 - |FN - FP| / (2TP + FP + FN)
                OutputDict.update(dict(volsim=VolumeSimilarity))
                
        return OutputDict