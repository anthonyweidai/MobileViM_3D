import math
from typing import Union
from numbers import Number

import torch
from torch import Tensor

from .baseMetric import AverageMetric
from .metricRegister import METRIC_REGISTRY

from .utils import makeConsistentDim


def computeMSE(
    Prediction: Tensor, Target: Tensor, 
    SpatialDims: int=2, Eps: float=1e-7, Mode=True
):
    """ Compute mean squared errors between two binary data, Mode for direct calling
    https://docs.monai.io/en/0.6.0/_modules/monai/metrics/regression.html#MSEMetric
    """
    Prediction2, Target2 = makeConsistentDim(Prediction, Target, SpatialDims)
    
    SquaredError = torch.pow(Target2 - Prediction2, exponent=2).float().mean()
    return SquaredError.cpu().numpy() if Mode else SquaredError


@METRIC_REGISTRY.register("error", "segmentation")
class MeanError(AverageMetric):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        """ RMSE: root mean squared error """
        self.Scale = self.opt.metric_scale

    def gatherMetrics(
        self, Prediction: Tensor, Target: Tensor, **kwargs,
    ) -> Union[Tensor, dict[str, Tensor]]:
        # no micro/macro difference
        OutputDict = dict()
        OutputDict["rmse"] = torch.sqrt(computeMSE(Prediction, Target, self.SpatialDims, Mode=False))
        return OutputDict
        
    def compute(self, **kwargs) -> dict[str, Number]:
        OutputDict = {
            "rmse": self.Value["rmse"] / math.ceil(self.NumTotal / self.LoaderBatch),
        }
        return OutputDict