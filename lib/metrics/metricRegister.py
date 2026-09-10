from ..utils import Registry


SEPARATOR = ":"
METRIC_REGISTRY = Registry(__file__, Separator=SEPARATOR, SkipFolder=[])


def getMetric(opt, MetricName: str, **kwargs):
    RegName = MetricName.lower() + SEPARATOR + opt.task
    return METRIC_REGISTRY[RegName](opt=opt, **kwargs)