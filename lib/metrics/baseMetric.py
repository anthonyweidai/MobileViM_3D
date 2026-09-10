import abc
import argparse
from numbers import Number
from typing import Any, List, Optional, Tuple, Union

import torch
from torch import Tensor

from ..utils import getLoaderBatch, tensor2PythonFloat


class BaseMetric(abc.ABC):
    def __init__(
        self,
        opt,
        Scale: Number=1.,
        Eps: Optional[float]=1e-7,
        IsTraining=True,
        **kwargs,
    ):
        self.opt = opt
        self.Eps = Eps
        self.Scale = Scale
        self.IsTraining = IsTraining
        
        self.Device = opt.device
        self.SpatialDims = opt.spatial_dims
        self.LoaderBatch = getLoaderBatch(opt)
        
        # We need the default value of device for tests.
        self.reset()
        
    @abc.abstractmethod
    def reset(self) -> None:
        """
        Resets all aggregated data.
        Called at the start of every epoch.
        """
        pass
    
    @abc.abstractmethod
    def update(
        self,
        Prediction: Union[Tensor, dict],
        Target: Union[Tensor, dict],
        Extras: dict[str, Any],
        BatchSize: Optional[int] = 1,
        **kwargs,
    ) -> None:
        """
        Processes a new batch of Prediction and Target for computing the metric.

        Args:
            Prediction: model outputs for the current batch
            Target: labels for the current batch
            Extras: dict containing extra information.
                During training this includes "loss" and "grad_norm" keys.
                During validaiton only includes "loss".
            BatchSize: optionally used to correctly compute the averages when
                the batch size varies across batches.
        """
        pass

    @abc.abstractmethod
    def compute(
        self,
    ) -> Union[Number, dict[str, Number]]:
        """
        Computes the metrics with the existing data.

        It gets called at every log iteration as well as the end of each epoch,
        e.g. train, val, valEMA.
        Logging happens at iteration 1 and every `common.log_freq` thereafter.

        Note: for computationally heavy metrics, you may want to increase `common.log_freq`.

        Returns:
            Depending on the metric, can return a scalar metric or a dictionary of metrics.
            Lists (or dicts of lists) are also generally accepted but not encouraged.
        """
        pass


class AverageMetric(BaseMetric):
    def reset(self):
        self.NumTotal = 0
        self.Value = None

    @abc.abstractmethod
    def gatherMetrics(
        self,
        Prediction: Union[Tensor, dict],
        Target: Union[Tensor, dict],
        Extras: dict[str, Any],
        *kwargs,
    ) -> Union[Tensor, dict[str, Tensor], Number]:
        raise NotImplementedError(
            "gatherMetrics needs to be implemented for subclasses of AverageMetric"
        )

    @torch.no_grad()
    def aggregateSum(self, Value: Union[Tensor, Number]) -> Union[float, List[float]]:
        """
        Given a Value, sums it up across distributed workers (if distributed) and
        returns the Value as a float (if scalar) or a Numpy array (otherwise).
        """
        return tensor2PythonFloat(Value)

    def update(
        self,
        Prediction: Union[Tensor, dict],
        Target: Union[Tensor, dict],
        Extras: Optional[dict[str, Any]]={},
        BatchSize: Optional[int]=1,
        **kwargs,
    ) -> None:
        Metric = self.gatherMetrics(
            Prediction=Prediction, Target=Target, Extras=Extras, BatchSize=BatchSize,
        )

        if isinstance(Metric, dict):
            # The values should be summed over all existing workers
            Metric = {k: self.aggregateSum(v) for k, v in Metric.items()}
            if self.Value is None:
                self.Value = Metric
            else:
                for k, v in Metric.items():
                    self.Value[k] += v
                    
        elif isinstance(Metric, (Tensor, Number)):
            if self.Value is None: self.Value = 0
            # The value should be summed over all existing workers
            self.Value += self.aggregateSum(Metric)
            
        else:
            raise ValueError(
                "gatherMetrics should return a Tensor or a dict \
                    containing Tensors. Got {}: {}".format(Metric.__class__, Metric)
            )
            
        # The count should be summed over all existing workers
        self.NumTotal += self.aggregateSum(BatchSize)
        
    def compute(self, **kwargs) -> Union[Number, dict[str, Number]]:
        # Count for val start point and divisor of 0
        if self.Value is None:
            return {}
        elif isinstance(self.Value, Number):
            return self.Value / max(1, self.NumTotal) * self.Scale
        elif isinstance(self.Value, dict):
            return {
                k: v / max(1, self.NumTotal) * self.Scale 
                for k, v in self.Value.items()
            }


class EpochMetric(BaseMetric):
    def __init__(
        self,
        opt: argparse.Namespace = None,
        Target: str = None,
        ForceCPU: bool = True,
    ):
        super().__init__(opt, Target)
        self.ForceCPU = ForceCPU

    def reset(self):
        self._predictions: List[Tensor] = []
        self._targets: List[Tensor] = []

    def update(
        self,
        Prediction: Union[Tensor, dict],
        Target: Union[Tensor, dict],
        Extras: dict[str, Any] = None,
        BatchSize: Optional[int] = 1,
    ) -> None:

        if not isinstance(Prediction, Tensor) or not isinstance(Target, Tensor):
            print(
                "EpochMetric only works on Tensor, got {} and {}.".format(
                    type(Prediction), type(Target)
                )
                + " Please set pred_key or target_key by setting the proper metric name:"
                + " `stats.val: ['metric_name(PredKey=key1, Target=key2)']`"
            )
            return
        
        Prediction = [Prediction]
        Target = [Target]

        # Detach the variables: we don't need to backprop in metrics
        Prediction = [x.detach() for x in Prediction]
        Target = [x.detach() for x in Target]
        # By default we move things to CPU so as to not put extra burden on GPU memory
        # but we allow child-classes/instances to keep the data on GPU for efficiency.
        if self.ForceCPU:
            Prediction = [x.cpu() for x in Prediction]
            Target = [x.cpu() for x in Target]

        self._predictions.extend(Prediction)
        self._targets.extend(Target)

    def getAggregates(self) -> Tuple[Tensor, Tensor]:
        """Aggregates Prediction and Target.

        This function gets called every time `self.compute` is called, which is at every
        log iteration as well as the end of each epoch, e.g. train, val, valEMA.
        Logging happens at iteration 1 and every `common.log_freq` thereafter.

        Note: for computationally heavy metrics, you may want to increase `common.log_freq`.
        """
        self._predictions = [torch.cat(self._predictions, dim=0)]
        self._targets = [torch.cat(self._targets, dim=0)]

        return self._predictions[0], self._targets[0]

    def computeWAggregates(self, Prediction: Tensor, Target: Tensor):
        """
        Computes the metrics given aggregated Prediction and Target.

        It gets called by `self.compute`. This happens at every
        log iteration as well as the end of each epoch, e.g. train, val, valEMA.
        Logging happens at iteration 1 and every `common.log_freq` thereafter.

        Note: for computationally heavy metrics, you may want to increase `common.log_freq`.
        """
        raise NotImplementedError

    def compute(self):
        Prediction, Target = self.getAggregates()
        return self.computeWAggregates(Prediction, Target)