from .baseDataset import SegBaseDataset
from .datasetRegister import DATASET_REGISTRY


@DATASET_REGISTRY.register("common", "segmentation")
class SegmentationDataset(SegBaseDataset):
    """ Dataset class for the PASCAL VOC 2012 dataset
    The structure of PASCAL VOC dataset should be something like this:
        PASCALVOC/mask
        PASCALVOC/train
        PASCALVOC/val
    """
    def __init__(self, opt, ImgPaths, Transform=None, **kwargs) -> None:
        super().__init__(opt, ImgPaths, Transform, **kwargs)
    
    def __getitem__(self, SizeNIndex) -> dict:
        Index = SizeNIndex if isinstance(SizeNIndex, int) else SizeNIndex[-1]
        ImgPath = self.ImgPaths[Index]
        Img = self.getImage(ImgPath)
        Mask = self.getImage(self.getMask(ImgPath), IsMask=True)
        
        Data = {"image": Img, "mask": Mask, "sample_id": Index}
        Data = self.Transform(Data)
        
        Mask = Data.pop("mask")
        Data["label"] = Mask if Mask is not None else 0
        return Data