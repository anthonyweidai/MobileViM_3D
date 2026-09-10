from .baseOptim import BaseOptim
from ..utils import Registry


OPTIM_REGISRY = Registry(__file__, BaseClass=BaseOptim, PurePathMode=True)


def buildOptimiser(NetParam, opt) -> BaseOptim:
    return OPTIM_REGISRY[opt.optim](opt, NetParam)