import math

from .utils import (
    deviceSetup, resizeShapeInit, 
    taskInitAfter, initPathMode,
)
from ..utils import (
    IN_CHANNELS, THREE_DIM_SETS, TaskAbbrevs, correctDatasetPath,
    getOnlyFolderNames, splitChecker
)

from ..utils import CLASS_NAMES


def segmentationParamsInit(opt):
    if not opt.setname:
        opt.aug_shape = 96
        opt.setname = "sample_seg" if opt.spatial_dims == 2 else "sample_seg_3d"
        
    SetName = opt.setname.lower()
    TestMode = not opt.setname or "sample_" in SetName
    
    opt.mask_fill = 255
    opt.ignore_idx = 255

    if not opt.class_names:
        opt.class_names = ["background"]
        if "sample_" in SetName:
            NameStr = "pascalanimal" if opt.spatial_dims == 2 else "curvas"
            opt.class_names.extend(CLASS_NAMES[NameStr])
        elif SetName:
            opt.class_names.extend(CLASS_NAMES[SetName])
            
    if "encoder_decoder" not in opt.seg_model_name:
        opt.seg_head_name = "-"
        opt.use_aux_head = False
        opt.model_name = opt.seg_model_name
    
    if not opt.use_aux_head:
        opt.aux_head_name = "-"
    
    # feature map guide
    if opt.seg_feature_guide:
        if opt.seg_feature_guide == 3:
            opt.fg_vit = 0
            opt.fg_link_vit = 0
            opt.fg_nostage5 = True
            
        if not opt.fg_use_guide:
            opt.fg_vit = 0
    
        if opt.fg_nostage5:
            opt.seg_head_name = "fg"
            opt.fg_for_head = True
            
    ## set se layer as None if does not use the whole module
    for n in ["fg_bottle", "fg_link_bottle"]:
        if getattr(opt, n) == 0:
            setattr(opt, "%s_se" % n, 0)
        
    if not opt.metric_indicator:
        opt.metric_indicator = "mdice"
    
    # test mode
    if TestMode:
        opt.epochs = 2
        opt.target_supplement = 0
        opt.target_exp = None
        opt.is_student = False
    
    if not opt.seg_loss_name:
      opt.seg_loss_name = opt.loss_name
    
    return opt


def overallParamsInit(opt):
    """Init before single task init"""
    # single init
    ## spatial dimensions
    if opt.setname and opt.setname.lower() in THREE_DIM_SETS: opt.spatial_dims = 3
    ## task single init
    opt = segmentationParamsInit(opt)
    SetName = opt.setname.lower()
  
  
    """Init after single task init"""
    # device
    opt = deviceSetup(opt)
    
    # dataset
    ## worker
    if opt.num_workers is not None and opt.num_workers < 1:
        opt.persistent_workers = False
    
    ## the number of splits
    _, opt.num_split = splitChecker(opt.setname, RSplit=True)
    
    ## path
    opt.custom_dataset_img_path, opt.dataset_path = \
        correctDatasetPath(opt.custom_dataset_img_path, opt.setname)

    ## input channels
    if opt.in_channels is None:
        opt.in_channels = IN_CHANNELS.get(SetName, IN_CHANNELS["default"])
        
    ## resize shape
    opt = resizeShapeInit(opt)

    ## get image path mode
    if not opt.get_path_mode:
        opt.get_path_mode = initPathMode(opt.dataset_path, SetName)
        
    ## class name
    if "traversal" in opt.class_names:
        opt.class_names = getOnlyFolderNames(opt.dataset_path + "/train")
        
    ## collate function name
    if not opt.collate_fn_name:
        opt.collate_fn_name = "default"

    ## exp level
    if opt.exp_level == "": # and not opt.lincls
        if "common" in opt.sup_method:
            # replace for sample dataset
            opt.exp_level = SetName.replace("/", "_")
            

    # train
    ## model
    opt.model_name = opt.model_name.lower()
    for k in TaskAbbrevs.values():
        if getattr(opt, "%s_model_name" % k) is None:
            setattr(opt, "%s_model_name" % k, opt.model_name)
    
    if not ("None" in opt.save_point or None in opt.save_point):
        opt.save_point = [int(i) for i in opt.save_point].sort()
    
    ## repetition
    if not opt.num_repeat:
        opt.num_repeat = opt.num_split
    
    ## start epoch
    if opt.val_start_epoch is None:
        opt.val_start_epoch = 0
  
    ## optimiser
    """ When using DDP+syncbn, bn is computated with a larger batch. 
    The learning rate should be tuned a bit higher (original_lr * num_gpus).
    https://discuss.pytorch.org/t/training-performance-degrades-with-distributeddataparallel/47152/22
    """
    if "cosine" in opt.schedular:
        if opt.lr is None:
            opt.lr = opt.lr_factor * opt.batch_size / 256
                
        if opt.warmup_init_lr is None:
            opt.warmup_init_lr = 50. * opt.lr
        
        if opt.max_lr is None:
            opt.max_lr = 10. * opt.lr
        
    if opt.lr is None:
        opt.lr = 2e-4
        opt.max_lr = 2e-3
    elif not opt.max_lr:
        opt.max_lr = 10 * opt.lr
    
    if opt.milestones is None:
        opt.milestones = math.ceil(0.1 * opt.epochs)
  
    if "adamw" in opt.optim:
        if opt.weight_decay is None:
            opt.weight_decay = 1.e-2
    
    ## number of views
    opt.views = 1
    # number of classes
    if opt.num_classes is None:
        opt.num_classes = len(opt.class_names)
    
    ## init task class numbers
    for k in TaskAbbrevs.values():
        if getattr(opt, "%s_num_classes" % k) is None:
            setattr(opt, "%s_num_classes" % k, opt.num_classes)
    
    # metrics
    opt.cls_loss_name = opt.loss_name
    
    ## task metrics
    if "common" not in opt.sup_method and not opt.selfsup_valid:
        opt.metric_indicator = "loss"
    
    opt.cls_num_classes = opt.num_classes
    ## top-k accuracy and supplementary metrics
    if opt.sup_method not in "common":
        opt.topk = (1, )
        opt.sup_metrics = False   
    elif opt.sup_metrics or opt.cls_num_classes < 10:
            opt.topk = (1, )
            opt.sup_metrics = True
    
    return taskInitAfter(opt)