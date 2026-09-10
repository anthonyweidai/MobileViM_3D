import numpy as np

import torch
from torch import Tensor


def makeConsistentDim(Prediction: Tensor, Target: Tensor, SpatialDims: int=2):
    # for segmentation
    if Prediction.ndim != Target.ndim:
        if Prediction.ndim == (SpatialDims + 2):
            Prediction = torch.argmax(Prediction, dim=1)
        elif Target.ndim == (SpatialDims + 2):
            Target = torch.argmax(Target, dim=1)
        else:
            NotImplementedError
    return Prediction, Target


def outDataTypeManage(Metric):
    # not used
    if isinstance(Metric, Tensor):
        Metric = Metric.cpu().numpy()
    return Metric


def manageMetricMM(
    MetricVal, UnionCount, 
    MetricsType="micro", Scale=1.0, Eps=1e-7,
):
    # manage the micro and macro types of metric
    if "macro" in MetricsType: # "macro"
        AvgVal = np.mean(MetricVal)
    else: # "micro"
        AvgVal = np.sum(MetricVal * UnionCount.T) / (np.sum(UnionCount) + Eps / Scale)
    return AvgVal * Scale


def applyRecursively(x, Rule, *args, **kwargs):
    if isinstance(x, dict):
        return {k: applyRecursively(v, Rule, *args, **kwargs) for k, v in x.items()}
    elif isinstance(x, (list, tuple)):
        return type(x)([applyRecursively(y, Rule, *args, **kwargs) for y in x])
    else:
        return Rule(x, *args, **kwargs)