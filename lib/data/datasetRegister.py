from .baseDataset import BaseDataset
from ..utils import Registry


SEPARATOR = ":"
DATASET_REGISTRY = Registry(__file__, BaseClass=BaseDataset, Separator=SEPARATOR, SkipFolder=['collate_fn', 'process', 'sampler'])


def getDataset(opt, ImgPaths, TargetSet=None, **kwargs) -> BaseDataset:
    if opt.reg_by_name:
        RegName = opt.setname.lower() + SEPARATOR + opt.task
    else:
        RegName = opt.sup_method + SEPARATOR + opt.task
    return DATASET_REGISTRY[RegName](opt, ImgPaths, TargetSet=TargetSet, **kwargs)