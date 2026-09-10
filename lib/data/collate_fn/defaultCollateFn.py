import numpy as np
from numbers import Number
from typing import List, Dict

import torch
from torch.utils.data import default_collate

from . import COLLATE_FN_REGISTRY


@COLLATE_FN_REGISTRY.register("official_default")
def pytorch_default_collate_fn(Batch, *args, **kwargs):
    """A wrapper around PyTorch's default collate function."""
    Batch = default_collate(Batch)
    return Batch


@COLLATE_FN_REGISTRY.register("default")
def defaultCollateFn(Batch: List[Dict], **kwargs):
    """Default collate function"""
    # get the keys for first element in the list,
    # assuming all elements have the same keys
    Keys = list(Batch[0].keys())
    NewBatch = {k: [] for k in Keys}
    
    for b in Batch:
        for k in Keys:
            NewBatch[k].append(b[k])

    # stack the Keys
    for k in Keys:
        BatchElements = NewBatch.pop(k)
        
        if isinstance(BatchElements[0], Number):
            # list of Number (includes int, float, np.int64, np.uint8)
            BatchElements = torch.as_tensor(BatchElements)
        elif isinstance(BatchElements[0], np.ndarray):
            if isinstance(BatchElements, (list, tuple)):
                # create a tensor from a list of np.ndarrays is slow
                BatchElements = np.array(BatchElements)
            BatchElements = torch.from_numpy(BatchElements)
        else:
            try:
                # stack tensors (including 0-dimensional)
                BatchElements = torch.stack(BatchElements, dim=0).contiguous()
            except Exception as e:
                print("Unable to stack the tensors. Error: {}".format(e))
                
        NewBatch[k] = BatchElements

    return NewBatch