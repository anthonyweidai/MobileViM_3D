from ...utils import Registry


COLLATE_FN_REGISTRY = Registry(__file__)


def buildCollateFn(opt, **kwargs):
    return COLLATE_FN_REGISTRY[opt.collate_fn_name]