from .baseScheduler import BaseLRScheduler
from ...utils import Registry


SCHEDULAR_REGISRY = Registry(__file__, BaseClass=BaseLRScheduler, PurePathMode=True)


def buildScheduler(opt, **kwargs) -> BaseLRScheduler:
    return SCHEDULAR_REGISRY[opt.schedular](opt, **kwargs)