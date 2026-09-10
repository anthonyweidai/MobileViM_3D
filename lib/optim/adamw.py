from torch.optim import AdamW

from . import OPTIM_REGISRY
from .baseOptim import BaseOptim


@OPTIM_REGISRY.register("adamw")
class AdamWOptimiser(BaseOptim, AdamW):
    """
    `AdamW <https://arxiv.org/abs/1711.05101>`_ optimiser
    """

    def __init__(self, opt, NetParam, **kwargs) -> None:
        BaseOptim.__init__(self, opt, **kwargs)
        AdamW.__init__(
            self,
            params=NetParam,
            lr=self.Lr,
            betas=(opt.beta1, opt.beta2),
            eps=self.Eps,
            weight_decay=opt.weight_decay,
            amsgrad=opt.amsgrad,
        )
