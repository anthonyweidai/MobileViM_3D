from .baseSegHead import BaseModule
from ....utils import Registry


SEGHEAD_REGISTRY = Registry(__file__, BaseClass=BaseModule, PurePathMode=True)


def buildSegmentationHead(opt, HeadName=None, **kwargs):
    if HeadName is None: HeadName = opt.seg_head_name
    return SEGHEAD_REGISTRY[HeadName](opt, **kwargs)