import numpy as np

from PIL import Image
import nibabel as nib
from nibabel.nifti1 import Nifti1Image


def readImagePil(ImgPath) -> Image.Image:
    try:
        Img = Image.open(ImgPath).convert("RGB")
    except:
        Img = None
    return Img


def readMaskPil(MaskPath):
    try:
        Mask = Image.open(MaskPath)
        if Mask.mode != "L" and Mask.mode != "P":
            print("Mask mode should be L or P. Got: {}".format(Mask.mode))
        return Mask
    except:
        return None


def readImageNii(ImgPath) -> Nifti1Image:
    # read nii.gz image file
    try:
        Img = nib.load(ImgPath)
    except:
        Img = None
    return Img


def readImage(ImgPath, SpatialDims=2, ReturnNii=False, IsMask=False):
    if SpatialDims == 2:
        return readImagePil(ImgPath) if not IsMask else readMaskPil(ImgPath)
    elif SpatialDims == 3:
        ImgVol = readImageNii(ImgPath)
        return ImgVol if ReturnNii \
            else ImgVol.get_fdata().astype(np.float32 if not IsMask else np.uint8)
    else:
        raise NotImplementedError("Unsopported spatial dimension %s" % SpatialDims)