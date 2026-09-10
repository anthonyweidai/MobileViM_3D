from typing import Iterator, Tuple

from . import SAMPLER_REGISTRY
from .baseSampler import BaseSampler
from ...utils import pair


@SAMPLER_REGISTRY.register("batch_sampler")
class BatchSampler(BaseSampler):
    """ Standard Batch Sampler for data parallel. 
    This sampler yields batches of fixed batch size and spatial resolutions.
    Args:
        opt: command line argument
        NumSamples: Number of samples in the dataset
        IsTraining: Training or validation mode. Default: False
    """

    def __init__(self, opt, NumSamples, IsTraining=False, *args, **kwargs) -> None:
        super().__init__(
            opt=opt, NumSamples=NumSamples, IsTraining=IsTraining
        )
        # spatial dimensions
        self.CropHeight, self.CropWidth = pair(opt.aug_shape)
        
    def __iter__(self) -> Iterator[Tuple[int, int, int]]:
        img_indices = self.get_indices()

        start_index = 0
        batch_size = self.batch_size_gpu0
        n_samples = len(img_indices)
        while start_index < n_samples:

            end_index = min(start_index + batch_size, n_samples)
            batch_ids = img_indices[start_index:end_index]
            start_index += batch_size

            if len(batch_ids) > 0:
                batch = [
                    (self.CropHeight, self.CropWidth, b_id) for b_id in batch_ids
                ]
                yield batch

    def extra_repr(self) -> str:
        extra_repr_str = super().extra_repr()
        extra_repr_str += (
            f"\n\tbase_im_size=(h={self.CropHeight}, w={self.CropWidth})"
            f"\n\tbase_batch_size={self.batch_size_gpu0}"
        )
        return extra_repr_str
