import math

from . import SCHEDULAR_REGISRY
from .baseScheduler import BaseLRScheduler


@SCHEDULAR_REGISRY.register("cosine")
class CosineScheduler(BaseLRScheduler):
    def __init__(self, opt, Optimiser, **kwargs) -> None:
        super().__init__(opt, Optimiser=Optimiser, **kwargs)
        self.EpochTemp = 0
        
        self.WarmupEpoches = opt.milestones
        if self.WarmupEpoches > 0:
            self.WarmupStep = (opt.max_lr - opt.warmup_init_lr) / self.WarmupEpoches

        self.Period = opt.epochs

    def getLR(self, Epoch: int) -> float:
        if Epoch == self.opt.milestones:
            self.EpochTemp = Epoch - 1
            self.WarmupEpoches += Epoch
            self.Period = self.opt.epochs
        
        if Epoch < self.WarmupEpoches:
            CurrLr = self.opt.warmup_init_lr + (Epoch - self.EpochTemp) * self.WarmupStep
        elif Epoch + 1 < self.Period:
            CurrLr = self.opt.lr + 0.5 * (self.opt.max_lr - self.opt.lr) * (1. + math.cos(math.pi * Epoch / self.Period))
        else:
            CurrLr = self.opt.lr
        return max(0.0, CurrLr)
    
    def step(self, Epoch: int):
        Values = self.getLR(Epoch)
        self.Optimiser.param_groups[0]["lr"] = Values
        
        self.LastLR = [group["lr"] for group in self.Optimiser.param_groups]