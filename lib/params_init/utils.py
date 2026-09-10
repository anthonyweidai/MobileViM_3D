import argparse

import torch
import torch.backends.cudnn as cudnn

from ..utils import (
    RESDICT, PRE_RESIZE, 
    getSysVersion, colourText, colourTextList,
    pair, getOnlyFolderNames, 
)


def deviceSetup(opt):
    opt.gpus_id = opt.gpus
    opt.gpus = [int(g) for g in opt.gpus.split(",")]
    
    getSysVersion(FullMode=False)
    
    NumGPUs = min(torch.cuda.device_count(), len(opt.gpus))
    if NumGPUs == 0 or (len(opt.gpus) == 1 and opt.gpus[0] == -1):
        Device = torch.device("cpu")
    elif NumGPUs == 1:
        Device = torch.device("cuda:" + str(opt.gpus[0]))
    else:
        Device = torch.device("cuda")
        if cudnn.is_available():
            cudnn.enabled = True
            cudnn.benchmark = False
            cudnn.deterministic = True
            # if isMaster(opt): print("CUDNN is enabled")
    
    print(
        "GPUs available in total: %s. Using %s with device id: [%s]." 
        % (
            colourText(str(NumGPUs)), colourText(Device.type), 
            ",".join(colourTextList(opt.gpus)),
        )
    )
    
    if NumGPUs > 0 and "cuda" in Device.type:
        assert torch.cuda.is_available(), "We need CUDA for training on GPUs."
                
    opt.start_rank = opt.rank
    
    if opt.world_size == -1:
        # Setting world-size the same as the number of available gpus
        opt.world_size = NumGPUs
        
    opt.device = Device
    opt.num_gpus = NumGPUs
    
    return opt


def taskInitAfter(opt):
    if "detection" in opt.task:
        # train
        if "sample" in opt.setname.lower():
            opt.val_start_epoch = 0
        else:
            opt.val_start_epoch = opt.save_point[-1] if "None" not in opt.save_point else 0
    
    return opt


def initPathMode(DatasetPath, SetName):
    # for param initialisation and norm counter
    FolderNames = getOnlyFolderNames(DatasetPath)
    if "split" in FolderNames or "split1" in FolderNames: # or use splitChecker in lib.utils
        PathMode = 1 # split number as subfolder
    elif any(s in SetName for s in [
        "cifar", "imagenet", "inaturalist", "monusac", "sports100",
    ]):
        PathMode = 3 # class name as subfolder
    else:
        PathMode = 4 # no subfolder
    return PathMode


def resizeShapeInit(opt):
    ## resize shape
    SetName = opt.setname.lower()
    if "resize" in SetName:
        DefaultResizeRes = RESDICT.get(SetName.replace("_resize", ""), RESDICT["default"][opt.task])
    else:
        DefaultResizeRes = RESDICT.get(SetName, RESDICT["default"][opt.task])
    
    if opt.aug_shape is None:
        opt.aug_shape = pair(DefaultResizeRes, RepNum=opt.spatial_dims)
    
    if all(v == DefaultResizeRes for v in pair(opt.aug_shape, RepNum=opt.spatial_dims)):
        opt.pre_resize = PRE_RESIZE.get(SetName, PRE_RESIZE["default"])
    else:
        opt.pre_resize = False
    return opt


def config2Options(opt: argparse.Namespace, Config: dict) -> argparse.Namespace:
    for k, v in Config.items():
        setattr(opt, k, v)
    return opt


def namespaceDiff(opt1, opt2):
    # show up differences between two argparse.Namespace
    for k in opt1.__dict__:
        if getattr(opt1, k) != getattr(opt2, k):
            print(k, getattr(opt1, k), getattr(opt2, k))