# from math import ceil
import multiprocessing

import torch

from .utils import colourText
from ..mathematics.utils import makeDivisible


def workerManager(opt, **kwargs):
    """ num_workers is not related to batch_size.
    num_workers > 0 is used to preprocess batches of data so that 
    the next batch is ready for use when the current batch has been finished. 
    More num_workers would consume more memory usage but is helpful to speed up the I/O process.
    https://discuss.pytorch.org/t/relation-between-num-workers-batch-size-and-epoch-in-dataloader/18201/2?
    Since larger the batch size, slower the training epoch. 
    The number of worker should be inverse variation to opt.batch_size.
    """
    # No of data workers = no of CPUs (if not specified or -1)
    NUM_PROC = multiprocessing.cpu_count()
    
    if opt.num_workers is None:
        # ml-cvnet implementation
        opt.num_workers = makeDivisible(
            NUM_PROC / max(torch.cuda.device_count(), 1) / opt.batch_size, 4,
        )
    
    # prefetch_factor option could only be specified in multiprocessing.
    if opt.num_workers <= 0:
        opt.prefetch_factor = None
    elif opt.prefetch_factor is None:
        opt.prefetch_factor = 2
        
    # pin_memory can speed up training in cuda device
    opt.pin_memory = True if "cuda" in opt.device.type and opt.pin_memory else False

    print("We use [%s/%d] workers, and pin memory is %s" \
        % (colourText(str(opt.num_workers)), 
            NUM_PROC, colourText(str(opt.pin_memory)))
    )
    return opt