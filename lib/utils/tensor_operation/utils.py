import numpy as np
from typing import Union

import torch
from torch import Tensor


def tensor2PythonFloat(Input) -> Union[int, float, np.ndarray]:
    if isinstance(Input, Tensor):
        if Input.numel() > 1:
            # For IOU, we get a C-dimensional tensor (C - number of classes)
            # so, we convert here to a numpy array
            return Input.cpu().numpy()
        elif hasattr(Input, "item"):
            return Input.item()
    elif isinstance(Input, (int, float, np.int64, np.ndarray)):
        return Input * 1.
    else:
        raise NotImplementedError(
            "The data type is not supported yet in tensor2PythonFloat function"
        )


def unsqueezeRight(x: torch.Tensor, n: int) -> torch.Tensor:
    """Unsqueeze multiple times in the rightmost dimension.
    https://github.com/PTB-MR/
    Example:
        tensor with shape (1,2,3) and n=2 would result in tensor with shape (1,2,3,1,1)
    Parameters
    ----------
    x
        tensor to unsqueeze
    n
        number of times to unsqueeze
    Returns
    -------
    unsqueezed tensor (view)
    """
    return x.reshape(*x.shape, *(n * (1,)))
