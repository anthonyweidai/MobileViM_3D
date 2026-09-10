from torch.utils.data.sampler import Sampler

from ...utils import Registry


SAMPLER_REGISTRY = Registry(__file__, BaseClass=Sampler, PurePathMode=True)


def buildSampler(opt, NumSamples: int, IsTraining: bool=False, **kwargs) -> Sampler:
    """
    Args:
    NumSamples: Number of data samples. It can be an integer specifying number of data samples for a given task
    or a mapping of task name and data samples per task in case of a chain sampler.
    """
    SamplerName = opt.sampler
    
    return SAMPLER_REGISTRY[SamplerName](opt, NumSamples=NumSamples, IsTraining=IsTraining, **kwargs)