import numpy as np
from typing import Optional
from functools import partial

from monai import transforms as mtransforms
from monai.utils import InterpolateMode as MInterpolateMode
from monai.transforms import (
    RandRotated, RandFlipd, Resized,
    ScaleIntensityRanged, NormalizeIntensityd,
)

from torch.utils.data import Dataset
import torchvision.transforms.functional as F

from .process import *
from .utils import initMeanStdByCsv, initIntensityRangeByCsv, all2OneMaks
from ..utils import (
    COLOUR_CODES, pair, colourText, makeDivisible,
    readImagePil, readMaskPil, readImageNii, readImage, getFileNameWithoutExt, 
)


class BaseDataset(Dataset):
    def __init__(
        self, opt, ImgPaths, 
        Transform=None, IsTraining: Optional[bool]=True,
        TargetSet=None, BBoxShift: int=10, **kwargs,
    ) -> None:
        super().__init__()
        self.opt = opt
        self.IsTraining = IsTraining
        self.ImgPaths =  np.asarray(ImgPaths)
        self.TargetSet = np.asarray(TargetSet) if TargetSet is not None else TargetSet
        
        self.ClassNames = opt.class_names
        self.InChannels = opt.in_channels
        self.SpatialDims = opt.spatial_dims
        self.RandAugProb = opt.random_aug_prob
        self.Size = pair(opt.aug_shape, RepNum=opt.spatial_dims)
        
        """ Mean and std should be the same for train and test dataset,
        use only meanstd of train set, test meanstd vary from task
        https://stats.stackexchange.com/questions/202287/
        """
        self.MeanStdType = self.IntenRangeType = "train"
        self.MeanStdDP = 6 # Decimal places
        
        self.readMaskPil = readMaskPil
        self.readImagePil = readImagePil
        self.readImageNii = readImageNii
        self.getImage = partial(readImage, SpatialDims=self.SpatialDims)
        
        self.initMeanStd()
        self.initIntensity()     
        self.initAugParams()       
        
        if self.SpatialDims == 2:
            self.InterpolationMode = F.InterpolationMode.BICUBIC
        else:
            # bicubic does not support 3D image yet
            self.InterpolationMode = MInterpolateMode.TRILINEAR
        
        if not Transform:
            if self.SpatialDims == 3:
                self.Transform = self.buildTransforms3D()
            else:
                raise NotImplementedError("Unsupported %dD data" % self.SpatialDims)

        print("The amount of original %s data: %s" % \
            ("train" if self.IsTraining else "test", colourText(str(self.__len__())))
        )

    def __getitem__(self, Index):
        pass

    def __len__(self):
        return len(self.ImgPaths)
    
    def getNumEachClass(self):
        pass
    
    def getSenFactor(self):
        pass

    def callerInit(self):
        pass
    
    def initAugParams(self):
        pass
    
    def buildTransforms(self):
        pass

    def buildTransforms3D(self):
        pass
    
    def initMeanStd(self):
        """ Normalization helps get data within a range and reduces 
        the skewness which helps learn faster and better.
        """
        self.MeanValues, self.StdValues = initMeanStdByCsv(
            self.opt, self.IsTraining, self.MeanStdType, self.MeanStdDP,
        )

    def initIntensity(self):
        self.IntensityRange = initIntensityRangeByCsv(
            self.opt, self.IsTraining, self.IntenRangeType,
        )


class SegBaseDataset(BaseDataset):
    def __init__(self, opt, ImgPaths, Transform=None, **kwargs) -> None:
        super().__init__(opt, ImgPaths, Transform, **kwargs)
        self.NumClasses = opt.seg_num_classes
        # mapping instructions to map BGR mask to classes
        self.Mapping = COLOUR_CODES.get(opt.setname, COLOUR_CODES["default"])
    
    def initAugParams(self):
        self.ShiftOffsets = makeDivisible(0.1 * self.MeanValues[0], 4) \
            if self.MeanValues is not None else 20

        self.EnCropSpatialSizes = [makeDivisible(round(i * 1.2), 8) for i in self.Size]
    
    def getMask(self, ImgPath):
        MaksFolder = "mask"
        ImgName = getFileNameWithoutExt(ImgPath)
        Ext = "nii.gz"
        
        MaskPath = "%s/%s/%s.%s" % (self.opt.dataset_path, MaksFolder, ImgName, Ext)
        return all2OneMaks(MaskPath, self.opt.setname)
    
    def buildTransforms3D(self):
        # Reference: SegVol, 3DSAM-adapter
        AugList = [
            # need clipping since intensity range is averaged values
            ChannelFirstd(["image", "mask"], self.InChannels, ChannelDim=None),
            ScaleIntensityRanged(
                keys=["image"],
                a_min=self.IntensityRange[0],
                a_max=self.IntensityRange[1],
                b_min=self.IntensityRange[0],
                b_max=self.IntensityRange[1],
                clip=True,
            ),
        ]
        if self.IsTraining:
            Coeff = 0.0174533 # np.pi / 180
            AugList.extend(
                [
                    CropForegroundd(
                        ["image", "mask"], self.IntensityRange[0], source_key="image",
                        start_coord_key=None, end_coord_key=None, 
                        channel_indices=None, allow_smaller=False,
                    ),
                    RandRotated(
                        ["image", "mask"], prob=self.RandAugProb,
                        range_x=10 * Coeff, range_y=10 * Coeff, 
                        range_z=10 * Coeff, keep_size=False,
                    ),
                    RandFlipd(["image", "mask"], prob=self.RandAugProb, spatial_axis=0),
                    RandFlipd(["image", "mask"], prob=self.RandAugProb, spatial_axis=1),
                    RandFlipd(["image", "mask"], prob=self.RandAugProb, spatial_axis=2),
                ]
            )
        
        AugList.extend([
            Resized(
                ["image", "mask"], self.Size, size_mode="all",
                mode=[self.InterpolationMode, "nearest"]
            ),
            SqueezeChanneld(["mask"], self.InChannels, ChannelDim=None),
        ])
        
        # normalise before converting to tensor
        if self.MeanValues:
            # MeanValues and StdValues are lists, and lengths are equal to channels.
            AugList.append(
                NormalizeIntensityd(
                    ["image"], channel_wise=True, # only impact on image
                    subtrahend=self.MeanValues, divisor=self.StdValues, 
                )
            )
        
        AugList.append(MetaTensor2Tensor()) # data values range within [-1, 1]
        return mtransforms.Compose(AugList)