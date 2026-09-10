from ..utils import TaskAbbrevs, Registry


SEPARATOR = ":"
# without a consistent base class
MODEL_REGISTRY = Registry(
    __file__, 
    Separator=SEPARATOR, 
    SkipFolder=[
        "layers", "modules",
        "base", "config", "anchor", "heads", "matcher", # higher level
    ]
)


def getModel(opt, **kwargs):
    RegName = getattr(opt, "%s_model_name" % TaskAbbrevs[opt.task])+ SEPARATOR + opt.task
    # encoder
    EncRegName = opt.model_name + SEPARATOR + "classification"
    Model = MODEL_REGISTRY[EncRegName](opt=opt, **kwargs)
    # decoder
    Model = MODEL_REGISTRY[RegName](opt=opt, Encoder=Model, **kwargs)
    return Model