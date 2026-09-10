from functools import partial

from monai.config import KeysCollection
from monai.transforms import CropForegroundd as MonaiCropForegroundd


def cfSelectFn(Img, MinIntensity):
    # select function of cropping foreground
    return Img > MinIntensity


class CropForegroundd(MonaiCropForegroundd):
    """ adapted to windows system
    Multiprocessing lib doesn't have it implemented on Windows
    https://discuss.pytorch.org/t/cant-pickle-local-object-dataloader-init-locals-lambda/31857/14?
    """
    def __init__(self, Keys: KeysCollection, MinIntensity: float=0, **kwargs):
        super().__init__(
            keys=Keys, 
            select_fn=partial(cfSelectFn, MinIntensity=MinIntensity), 
            **kwargs,
        )