import math
import numpy as np
from numbers import Number
from typing import Any, Union

import torch
from torch import Tensor

from .baseMetric import AverageMetric
from .metricRegister import METRIC_REGISTRY


@METRIC_REGISTRY.register("loss", "classification")
@METRIC_REGISTRY.register("loss", "segmentation")
@METRIC_REGISTRY.register("loss", "detection")
@METRIC_REGISTRY.register("loss", "regression")
@METRIC_REGISTRY.register("loss", "autoencoder")
class LossMetric(AverageMetric):
    def formatManager(self, Loss: Union[Tensor, Number], **kwargs) -> Tensor:
        if isinstance(Loss, Tensor):
            return Loss
        elif isinstance(Loss, (Number, np.ndarray)):
            return torch.tensor(Loss, device=self.Device)
        else:
            raise NotImplementedError(f"Loss metric do not supports {Loss} with {type(Loss)} type.")
    
    def gatherMetrics(self, Extras: dict[str, Any], **kwargs) -> Union[Tensor, dict[Tensor]]:
        """
        This function gather losses from different processes and converts to float.
        """
        if Extras is None: Extras = {}
        Loss = Extras.get("loss", None)
        if Loss is None: Loss = 0.0

        if isinstance(Loss, dict):
            for k, v in Loss.items():
                Loss[k] = self.formatManager(v)
        else:
            Loss = self.formatManager(Loss)
            
        return dict(loss=Loss)
    
    def compute(self, **kwargs) -> dict[str, Number]:
        if self.opt.loss_reduction == "mean":
            OutputDict = {"loss": self.Value["loss"] / math.ceil(self.NumTotal / self.LoaderBatch)} 
        else:
            OutputDict = {"loss": self.Value["loss"] / self.NumTotal} 
        return OutputDict