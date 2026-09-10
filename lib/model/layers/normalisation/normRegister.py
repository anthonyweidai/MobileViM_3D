from .baseNorm import BaseNorm
from ....utils import Registry


NORM_LAYER_REGISTRY = Registry(__file__, BaseClass=BaseNorm, PurePathMode=True)


def buildNormLayer(opt, NormType=None):
    NormType = NormType or opt.norm_layer
    if "sync_" not in NormType:
        for n in ["batch_norm", "instance_norm"]:
            if n in NormType:
                NormType = "%s_%dd" % (n, opt.spatial_dims)
                break
    elif "cuda" not in opt.device.type:
        NormType = "batch_norm_%dd" % (opt.spatial_dims)
    
    return NORM_LAYER_REGISTRY[NormType]