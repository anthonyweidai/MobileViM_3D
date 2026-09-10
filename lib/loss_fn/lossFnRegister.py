from ..utils import Registry


SEPARATOR = ":"
LOSS_FN_REGISTRY = Registry(__file__, Separator=SEPARATOR, SkipFolder=[])


def getLossFn(opt, Task=None, **kwargs):
    if not Task:
        Task = opt.task
    
    RegName = opt.loss_name + SEPARATOR + opt.task
    
    return LOSS_FN_REGISTRY[RegName](opt, Task=Task, **kwargs)