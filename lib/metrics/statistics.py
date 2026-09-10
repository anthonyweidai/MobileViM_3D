import traceback
import numpy as np
import pandas as pd

import torch
from torch import Tensor

from .baseMetric import BaseMetric
from .metricRegister import getMetric
from .metricRegister import METRIC_REGISTRY
from .utils import applyRecursively, manageMetricMM
from ..loss_fn.utils import getAuxOut


class Statistics(object):
    def __init__(self, opt, IsTraining=True) -> None:
        self.opt = opt
        self.IsTraining = IsTraining
        
        # init saved metrics
        self.initMetrics()
        self.epochParamsInit()
    
    def initMetrics(self):
        # use metric list name for initialisation in write running metrics
        self.Decimal = 8 # rounding decimal
        self.Scale = self.opt.metric_scale
        self.ScaleOneNames = [] # "VolSim"
        
        self.Epoch = 0
        self.ComMetric = 0 # for save the best model
        self.DetValStart = False # for coco detection
        self.Flag = not self.IsTraining and self.opt.sup_method in "common"
        
        MetricNames = []
        AvgMetricNames = [] # micro/macro version
        MetricListNames = ["Loss"] # for data saving
        
        self.UsedMetrics = ["loss"] # for registered metrics
        # Jaccard/IoU, F1 score, dice score
        MetricNames.extend(["mIoU", "mDice"])
        if self.Flag:
            # E-measure, mean absolute errors, surface distance
            MetricListNames.extend(["RMSE"])
            # segmentation confusion matrix
            AvgMetricNames.extend(["Recall", "Precision", "F1Score", "F2Score"])
        
        # for registered metrics
        self.UsedMetrics.extend(["mask_region", "error"]) 

        MetricNames.extend(["Avg" + m for m in AvgMetricNames])
        self.AvgMetricNames = AvgMetricNames
        
        ## create multiple empty lists
        # MetricListNames.extend(MetricNames)
        MetricListNames[1:1] = MetricNames
        for n in MetricListNames:
            setattr(self, n + "List", [])
        self.MetricListNames = MetricListNames
            
        ## create multiple pandas frames, save time in training
        self.MetricDfNames = []
        for n in MetricNames:
            # remove "m" and "Avg" in the substring head
            if "m" == n[0] and n[1].isupper():
                n = n[1:]
            n = n.replace("Avg", "")
            self.MetricDfNames.append(n)
            setattr(self, n + "Df", pd.DataFrame())

        # key is the metric name and value is the value
        self.MetricDict: dict[str, BaseMetric] = {}
        RegisteredMetrics = [
            m.replace(":%s" % self.opt.task, "") 
            for m in list(METRIC_REGISTRY.keys()) if  self.opt.task in m
        ]
        for m in self.UsedMetrics:
            if m in RegisteredMetrics:
                self.MetricDict[m] = getMetric(
                    self.opt, m, IsTraining=self.IsTraining, IoUTypes=["bbox"]
                )

    def epochParamsInit(self):
        # reset metric values for each epoch
        for v in self.MetricDict.values(): v.reset()
       
    @torch.no_grad()
    def update(self, PredLabel, TargetLabel: Tensor, Loss: Tensor, Epoch: int):
        self.Epoch = Epoch
        PredLabel, _ = getAuxOut(PredLabel)
        # Could be unequal to the default batchsize
        if isinstance(PredLabel, Tensor):
            # vanilla tensor shape
            BatchSize = PredLabel.shape[0] 
        elif isinstance(PredLabel, dict):
            # simsiam takes 2 view of same image as one pair in the batch
            BatchSize = list(PredLabel.values())[0].shape[0] 
        elif isinstance(PredLabel, list):
            # for barlow
            BatchSize = PredLabel[0].shape[0]
        else:
            # BatchSize = self.opt.batch_size
            raise NotImplementedError
                    
        # calculating metric values
        Rule = lambda x: x.detach().clone() if isinstance(x, Tensor) else x
        for n, m in self.MetricDict.items():
            Extra = {"loss": applyRecursively(Loss, Rule)} if "loss" in n else None
            try:
                m.update(
                    Prediction=applyRecursively(PredLabel, Rule),
                    Target=applyRecursively(TargetLabel, Rule),
                    Extras=Extra,
                    BatchSize=BatchSize,
                )
            except Exception as e:
                traceback.print_exc()
                print("Caught an error while updating metric {}: {}".format(n, e))
        
    @torch.no_grad()
    def averageMetrics(self):
        """ Computes average statistics of all metrics and returns them as a list of strings.
        Examples:
        "ae": 0.003
        {"iou": 0.88, "dice": 0.89, "recall": 0.86, ...}
        Use dictionary to save list value and then save it in csv, since one by one line saving is slow
        """
        MetricsStats = {}
        for n, m in self.MetricDict.items():
            Metrics = applyRecursively(
                m.compute(), # Epoch=self.Epoch
                lambda x: np.round(x * 1.0, decimals=self.Decimal),
            )
            if isinstance(Metrics, dict):
                MetricsStats.update(Metrics)
            else:
                MetricsStats[n] = Metrics
        
        ComMetricName = "Loss"
        MetricNames = {"Loss": "loss"}
        
        MeanMetricNames = []
        if len(MetricsStats) > 1:
            MeanMetricNames = ["IoU", "Dice"]
            if self.Flag:
                MetricNames.update({"RMSE": "rmse"})
            ComMetricName = "mDice"
            
        # compute and assign mean and average values
        for m, Names in zip(["m", "Avg"], [MeanMetricNames, self.AvgMetricNames]):
            for n in Names:
                Scale = self.Scale if n not in self.ScaleOneNames else 1
                setattr(
                    self, m + n, manageMetricMM(
                        MetricsStats[n.lower()], MetricsStats["count"], 
                        self.opt.metric_type, Scale,
                    )
                )
        
        # assign final output metrics to self
        for k, v in MetricNames.items():
            setattr(self, k, MetricsStats[v])
        self.ComMetric = getattr(self, ComMetricName)
        
        # assign list
        for n in self.MetricListNames:
            # update dataframe values
            Value = getattr(self, n)
            setattr(
                self, n + "List", getattr(self, n + "List") + [Value]
            )
        # assign dataframe
        for n in self.MetricDfNames:
            # update dataframe values
            Value = MetricsStats[n.lower()]
            setattr(
                self, n + "Df", pd.concat(
                    [getattr(self, n + "Df"), pd.DataFrame(Value).T]
                )
            )