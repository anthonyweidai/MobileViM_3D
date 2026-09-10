import copy
import math
import random
import argparse
import numpy as np
from typing import Any, Iterator, List, Tuple

import torch
from torch.utils.data.sampler import Sampler


class BaseSampler(Sampler):
    """ Base class for standard and DataParallel Sampler.
    Every subclass should implement `__iter__` method, providing a way to iterate
    over indices of dataset elements.
    """
    def __init__(
        self,
        opt: argparse.Namespace,
        NumSamples: int,
        LoaderBatch: int=None,
        IsTraining: bool=False,
        *args,
        **kwargs,
    ) -> None:
        """
        Args:
            opt: command line argument
            NumSamples: Number of samples in the dataset
            IsTraining: Training mode or not. Default: False
        """
        self.opt = opt
        self.NumSamples = NumSamples
        self.IsTraining = IsTraining
        
        # max between 1 and number of available GPUs. 1 because for supporting CPUs
        n_gpus: int = max(1, torch.cuda.device_count())
        Ids, TotalSize, _ = self.getAllIndices(n_gpus)

        self.img_indices = Ids
        self.n_samples = TotalSize
        # Handled batch size inside data sampler
        self.batch_size_gpu0 = opt.batch_size if LoaderBatch is None else LoaderBatch
        self.NumUnits = n_gpus
        self.Epoch = 0
        
        self.Shuffle = True if IsTraining and opt.loader_shuffle else False
        # unused, load data without using ordered data?
        self.DropLast = opt.drop_last

        self.num_repeats = 1
        self.trunc_rep_aug = False
        if IsTraining:
            # enable these arguments for repeated data augmentation
            # https://openaccess.thecvf.com/content_CVPR_2020/papers/Hoffer_Augment_Your_Batch_Improving_Generalization_Through_Instance_Repetition_CVPR_2020_paper.pdf
            self.num_repeats = opt.sampler_num_repeats
            self.trunc_rep_aug = opt.sampler_truncated_aug

    def getAllIndices(self, Divisor):
        NumSamplesPerUnit = int(math.ceil(self.NumSamples * 1.0 / Divisor))
        TotalSize = NumSamplesPerUnit * Divisor
        
        Ids = [idx for idx in range(self.NumSamples)]
        # This ensures that we can divide the batches evenly across GPUs
        Ids += Ids[: (TotalSize - self.NumSamples)]
        assert TotalSize == len(Ids)

        return Ids, TotalSize, NumSamplesPerUnit
    
    def get_indices(self) -> List[int]:
        """Returns a list of indices of dataset elements to iterate over.

        ...note:
            If repeated augmentation is enabled, then indices will be repeated.
        """
        img_indices = copy.deepcopy(self.img_indices)
        if self.Shuffle:
            random.seed(self.Epoch)
            random.shuffle(img_indices)

            if self.num_repeats > 1:
                # Apply repeated augmentation
                """Assume that we have [0, 1, 2, 3] samples. With repeated augmentation,
                we first repeat the samples [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3] and then select 4
                samples [0, 0, 0, 1]. Note that we do shuffle at the beginning, so samples are not the
                same at every iteration.
                """
                n_samples_before_repeat = len(img_indices)
                img_indices = np.repeat(img_indices, repeats=self.num_repeats)
                img_indices = list(img_indices)
                if self.trunc_rep_aug:
                    img_indices = img_indices[:n_samples_before_repeat]
        return img_indices

    def __iter__(self) -> Iterator[Tuple[Any, ...]]:
        pass

    def __len__(self) -> int:
        return len(self.img_indices) * (1 if self.trunc_rep_aug else self.num_repeats)
    
    def setEpoch(self, Epoch: int) -> None:
        """Helper function to set epoch in each sampler."""
        self.Epoch = Epoch

    def updateScales(self, *args, **kwargs) -> None:
        """Helper function to update scales in each sampler. This is typically useful in variable-batch sampler.

        Subclass is expected to implement this function. By default, we do not do anything
        """

    def update_indices(self, new_indices: List[int]) -> None:
        """Update indices to new indices. This function might be useful for sample-efficient training."""
        self.img_indices = new_indices
        
    def epochInit(self, Epoch: int, *args, **kwargs) -> None:
        # only needed to be applied in training each epoch
        self.setEpoch(Epoch)
        self.updateScales(*args, **kwargs)

    def extra_repr(self) -> str:
        extra_repr_str = (
            f"\n\t num_repeat={self.num_repeats}"
            f"\n\t trunc_rep_aug={self.trunc_rep_aug}"
        )
        return extra_repr_str

    def __repr__(self) -> str:
        return "{}({}\n)".format(self.__class__.__name__, self.extra_repr())