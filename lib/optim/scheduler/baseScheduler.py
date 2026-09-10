class BaseLRScheduler(object):
    def __init__(self, opt, Optimiser=None, Eps=1e-8, **kwargs) -> None:
        super().__init__()
        self.opt = opt
        self.Eps = Eps
        self.BaseLR = opt.lr
        self.MaxLR = opt.max_lr
        self.Optimiser = Optimiser
        self.Milestones = opt.milestones
        
        self.LastLR = None
        self._last_lr = self.LastLR # adpated to pytorch default variable

    def getLR(self, Epoch: int, CurrIter: int):
        pass
    
    def step(self, Epoch: int):
        pass

    def getLastLR(self):
        # Return last computed learning rate by current scheduler.
        return self.LastLR