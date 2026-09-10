class BaseOptim(object):
    def __init__(self, opt, **kwargs) -> None:
        self.Eps = 1e-8
        self.Lr = opt.lr
        self.WeightDecay = opt.weight_decay
