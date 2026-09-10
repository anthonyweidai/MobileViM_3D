# System
TextColors = {
    "end_colour": "\033[0m",
    "bold": "\033[1m", # 033 is the escape code and 1 is the color code 
    "error": "\033[31m", # red
    "light_green": "\033[32m",
    "light_yellow": "\033[33m",
    "light_blue": "\033[34m",
    "light_cyan": "\033[36m",
    "warning": "\033[37m", # white
}

# Task
TaskAbbrevs = {"segmentation": "seg"}


# Metric
LogMetrics = {
    "3d": ["accuracy", "mdice", "map"],
}
WeightMetrics = {
    "segmentation_3d": "maxdice",
}


# Dataset
## 3D dataset names 
THREE_DIM_SETS = [
    "sample_seg_3d", "atlas", "toothfairy2", "pengwin", "brats2024t3",
]

## input channels
IN_CHANNELS = {
    "default": 3,
    **dict.fromkeys(THREE_DIM_SETS + [], 1)
}

## spatial index for 3D data
SPATIAL_IDS = {
    # torchvision
    "1D": {"default": [0]},
    "2D": {"default": [0, 1]}, 
    # monai
    "3D": {
        "default": [2, 1, 0],
    },
}

## resolution
RESDICT = {
    "default": {
        "classification" : 224, "segmentation": 512, 
    }, # "regression"
    # segmentation
    **dict.fromkeys(
        [
            "atlas", "toothfairy2", "brats2024t3", "pengwin",
        ], 128
    ),
}

## True if using the default image size of dataset, otherwise False
PRE_RESIZE = {
    "default": False,
    **dict.fromkeys(
        [
            # 3D segmentation
            "atlas", "toothfairy2", "pengwin", "brats2024t3",
        ], False
    ),
}

## class names
CLASS_NAMES = {
    **dict.fromkeys(["atlas"], ["liver", "tumour"]),
    "pengwin": ["sacrum", "left hipbone", "right hipbone"],
    "toothfairy2": [
        "Lower Jawbone", "Upper Jawbone", "Left Inferior Alveolar Canal", "Right Inferior Alveolar Canal", "Left Maxillary Sinus", 
        "Right Maxillary Sinus", "Pharynx", "Bridge", "Crown", "Implant", 
        "Upper Right Central Incisor", "Upper Right Lateral Incisor", "Upper Right Canine", "Upper Right First Premolar", "Upper Right Second Premolar", 
        "Upper Right First Molar", "Upper Right Second Molar", "Upper Right Third Molar (Wisdom Tooth)", # , , 
        "Upper Left Central Incisor", "Upper Left Lateral Incisor", "Upper Left Canine", "Upper Left First Premolar", "Upper Left Second Premolar", 
        "Upper Left First Molar", "Upper Left Second Molar", "Upper Left Third Molar (Wisdom Tooth)", # , , 
        "Lower Left Central Incisor", "Lower Left Lateral Incisor", "Lower Left Canine", "Lower Left First Premolar", "Lower Left Second Premolar", 
        "Lower Left First Molar", "Lower Left Second Molar", "Lower Left Third Molar (Wisdom Tooth)", # , "NA", 
        "Lower Right Central Incisor", "Lower Right Lateral Incisor", "Lower Right Canine", "Lower Right First Premolar", "Lower Right Second Premolar", 
        "Lower Right First Molar", "Lower Right Second Molar", "Lower Right Third Molar (Wisdom Tooth)", # , , 
    ], # too low performance (<12% mIoU)
    "brats2024t3": ["GTV"], # gross tumor volume
}