import argparse


def argumentsTask(Parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    Group = Parser.add_argument_group(title="task arguments")
    
    # basic experiment setting
    Group.add_argument(
        "--task", 
        default="segmentation",
        choices=(
            "segmentation", 
        ), 
        help="Choose the deep learning task",
    )
    """
    Most contrastive learning (CLR) methods are used in 
    classification datasets (with rich number of images).
    So, we set the task of CLR is classification.
    """
    Group.add_argument(
        "--sup_method", 
        type=str, 
        default="common",
        choices=(
            "common",
        ),
        help="Choose the supervison method",
    )

    return Parser


def argumentsDataset(Parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    Group = Parser.add_argument_group(title="dataset arguments")
    
    Group.add_argument(
        "--custom_dataset_img_path", 
        default="./dataset/",
        help="Custom dataset",
    )
    Group.add_argument(
        "--setname", 
        default="", # "sample/split",
        help="See lib/dataset/ for available datasets",
    )
    Group.add_argument(
        "--reg_by_name", 
        action="store_true",
        help="Get dataset by its name but not supervision method?",
    )
    Group.add_argument(
        "--num_split",
        type=int, 
        default=1,
        help="The number of cross-validation folders",
    ) # k-folds validation
    Group.add_argument(
        "--get_path_mode", 
        type=int,
        default=None,
        help="The mode of calling images from their path",
    )
    Group.add_argument(
        "--num_repeat", 
        type=int, 
        default=1,
        help="The number of repeated split",
    ) # k-folds validation
    Group.add_argument(
        "--in_channels",
        type=int,
        default=None,
        help="The input channels of data",
    )
    Group.add_argument(
        "--spatial_dims",
        type=int,
        default=3,
        help="The spatial dimensions of input data, defaulted as 2 for 2D image",
    )
    Group.add_argument(
        "--aug_shape",
        nargs="+",
        type=int,
        default=None,
        help="The resize/crop scale (h, w) of transforms during training",
    )
    Group.add_argument(
        "--pre_resize", 
        action="store_true",
        help="Is the dataset aleady pre-processed?",
    )
    Group.add_argument(
        "--mask_fill", 
        type=int, 
        default=255,
        help="Fill padded region with a value",
    )
    Group.add_argument(
        "--class_names", 
        type=list, 
        default=None,
        help="The list of class name",
    )
    Group.add_argument(
        "--drop_last", 
        action="store_true",
        help="Drop last in data loader, due to batch normlisation, \
            the value will be different for the same dataset for different batch size",
    )
    Group.add_argument(
        "--loader_shuffle", 
        action="store_false",
        help="Shuffle data order while loading?",
    )
    Group.add_argument(
        "--use_meanstd", 
        action="store_false",
        help="Use mean std to process the image RGB channels? Should be true in inferencing.",
    )
    Group.add_argument(
        "--use_intensity_range", 
        action="store_false",
        help="Use intensity range clip to process the 3D image? Should be true in inferencing.",
    )
    
    return Parser


def argumentsSampler(Parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    Group = Parser.add_argument_group(title="sampler arguments")
    
    # base sampler
    Group.add_argument(
        "--sampler",
        type=str,
        default="batch_sampler",
        help="Name of the sampler. Defaults to batch sampler.",
    )
    Group.add_argument(
        "--sampler_num_repeats",
        type=int,
        default=1,
        help="Repeat the training dataset samples by this factor in each epoch (aka repeated augmentation). "
        "This effectively increases samples per epoch. As an example, if dataset has 10000 samples "
        "and sampler.num_repeats is set to 2, then total samples in each epoch would be 20000. "
        "Defaults to 1.",
    )

    Group.add_argument(
        "--sampler_truncated_aug",
        action="store_true",
        help="When enabled, it restricts the sampler to load a subset of the training dataset such that"
        "number of samples obtained after repetition are the same as the original dataset."
        "As an example, if dataset has 10000 samples, sampler.num_repeats is set to 2, and "
        "sampler.truncated_repeat_aug_sampler is enabled, then the sampler would sample "
        "10000 samples in each epoch. Defaults to False.",
    )
    
    # base distributed data-parallel sampler
    Group.add_argument(
        "--use_shards",
        action="store_true",
        help="Use data sharding. Only applicable to DDP. Defaults to False.",
    )
    Group.add_argument(
        "--disable_shards_shuffle",
        action="store_true",
        help="Disable shuffling while sharding for extremely large datasets. Defaults to False.",
    )

    return Parser


def argumentsAugmentation(Parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    Group = Parser.add_argument_group(title="data augmentation arguments")
    
    Group.add_argument(
        "--mask_ratio", 
        type=float, 
        default=None,
        help="The mask ratio of masking image in data loading",
    )
    Group.add_argument(
        "--random_aug_order", 
        action="store_true", 
        help="Use random order for augmentation methods?",
    )
    Group.add_argument(
        "--auglikeclr", 
        action="store_true", 
        help="Use contrastive learning augmentation in classification",
    )
    Group.add_argument(
        "--random_aug_prob", 
        type=float, 
        default=0.5,
        help="The probability for random augmentation",
    )
    
    return Parser


def argumentsCollateFn(Parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    Group = Parser.add_argument_group(title="collate function arguments")
    
    Group.add_argument(
        "--collate_fn_name",
        type=str, 
        default=None,
        help="The collate function name",
    )
    return Parser


def argumentsModeling(Parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    Group = Parser.add_argument_group(title="modeling arguments")
    
    # task
    Group.add_argument(
        "--num_classes", 
        type=int, 
        default=None,
        help="The number of classes",
    )
    Group.add_argument(
        "--init_weight", 
        action="store_false", 
        help="Apply weight initialisation?",
    )
    Group.add_argument(
        "--model_name", 
        type=str,
        default="mobilevimxxs",
        help="Model architecture",
    )
    
    # drop rate
    Group.add_argument(
        "--init_drop_rate1", 
        type=float, 
        default=0,
        help="Initial drop rate for inner module",
    )
    Group.add_argument(
        "--init_drop_rate2", 
        type=float, 
        default=0,
        help="Initial drop rate for fuse module",
    )
    
    ## classification
    Group.add_argument(
        "--cls_model_name", 
        type=str,
        default=None,
        help="Classification model architecture, used in getModel",
    )
    """ resnet18, mobilenetv3_small, mae_vit_base_patch16 """
    Group.add_argument(
        "--cls_num_classes", 
        type=int, 
        default=None,
        help="The number of classes",
    )
    ### mamba
    Group.add_argument(
        "--use_mamba", 
        action="store_true", 
    )
    Group.add_argument(
        "--mamba_dimin", 
        type=int, 
        default=0,
        help="Dimension-independent mamba \
            0 w/o | 1 separate before mamba | 2 separate within mamba (not implemented)",
    )
    Group.add_argument(
        "--mamba_dualdirection", 
        action="store_true", 
        help="Stack DP-Mamba output before output projection",
    )
    Group.add_argument(
        "--d_state", 
        type=int, 
        default=16,
        help="d_state in mamba",
    )
    Group.add_argument(
        "--mamba_block", 
        type=int, 
        default=1,
        choices=(1, 2, 3, 4),
        help="Mamba block version",
    )
    Group.add_argument(
        "--mamba_se", 
        type=int, 
        default=0,
        help="0 none | 1 squeeze and excitation | 2 monte carlo attention \
            | 3 same channel attention | 4 CBAM | 5 coordinate attention",
    )
    ### mamba default
    Group.add_argument(
        "--use_fp32_residual", 
        action="store_false", 
        help="Student network for model disillation",
    )
    ### not useful
    Group.add_argument(
        "--mamba_dimin_expand",
        action="store_true", 
        help="Use expansion in DP mode 1",
    )
    Group.add_argument(
        "--mamba_catproj", 
        action="store_true", 
        help="Concat the output from mamba and project them",
    )

    ## segmentation
    Group.add_argument(
        "--seg_num_classes", 
        type=int, 
        default=None,
        help="The number of classes",
    )
    Group.add_argument(
        "--seg_model_name", 
        type=str, 
        default="encoder_decoder",
        help="Segmentation models\' name",
    )                         
    Group.add_argument(
        "--seg_head_name", 
        type=str, 
        default="fg",
        help="Segmentation heads\' name",
    )
    Group.add_argument(
        "--use_aux_head", 
        action="store_true",
        help="Do you use auxiliary head during segmentation?",
    )
    Group.add_argument(
        "--aux_head_name", 
        type=str, 
        default="auxhead",
        help="Auxiliary heads\' name",
    )
    # deeplabv3
    Group.add_argument(
        "--use_sep_conv", 
        action="store_true",
        help="Use seperable convolution in deeplabv3?",
    ) 
    ### feature guidance module
    Group.add_argument(
        "--seg_feature_guide", 
        type=int, 
        default=3,
        help="0 not used | 1 standard | 2 lightweight | 3 lightweight + skip bt",
    )
    Group.add_argument(
        "--fg_start_stage", 
        type=int, 
        default=1,
        help="The starting stage of feature map guide",
    )
    Group.add_argument(
        "--fg_resize_stage", 
        type=int, 
        default=0,
        help="The stage for resizing feature maps before concatenation 0 | -1, \
            0 has better result, but may be slower in some heads",
    )
    Group.add_argument(
        "--fg_bottle", 
        type=int, 
        default=1,
        help="0 None | 1 default | 2 bottleneck",
    )
    Group.add_argument(
        "--fg_bottle_se", 
        type=int, 
        default=0,
        help="0 none | 1 squeeze and excitation | 2 monte carlo attention \
            | 3 CBAM | 4 coordinate attention",
    )
    Group.add_argument(
        "--fg_use_guide", 
        action="store_false",
        help="Use feature guide (cslayer)?",
    )
    Group.add_argument(
        "--moc_order", 
        action="store_false",
        help="The order of moc attention in fg bottleneck",
    )
    Group.add_argument(
        "--fg_link", 
        type=int, 
        default=2,
        help="0 None | 1 link head | 2 catlink head",
    )
    Group.add_argument(
        "--fg_for_head", 
        action="store_true",
        help="Use the feature map guide the segmentation head? Only useful in fg_nostage5",
    )
    ### tested modules
    Group.add_argument(
        "--link_expansion", 
        type=float, 
        default=0.25,
        help="the expansion rate for up link in fg3",
    )
    ### low-efficacy modules
    Group.add_argument(
        "--fg_link_bottle", 
        type=int, 
        default=0,
        help="0 None | 1 default | 2 bottleneck",
    )
    Group.add_argument(
        "--fg_link_bottle_se", 
        type=int, 
        default=2,
        help="0 none | 1 squeeze and excitation | 2 monte carlo attention \
            | 3 CBAM | 4 coordinate attention",
    )
    Group.add_argument(
        "--fg_cat_shuffle", 
        action="store_true",
        help="Shuffle the concatenated feature maps?",
    )
    ### low-performance modules
    Group.add_argument(
        "--fg_nostage5", 
        action="store_true",
        help="Remove stage-5 layers in encoder?",
    )
    
    
    # module
    ## norm layer
    Group.add_argument(
        "--norm_layer", 
        type=str, 
        default="batch_norm",
        help="Normalisation layer name",
    )

    return Parser


def argumentsLoss(Parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    Group = Parser.add_argument_group(title="loss arguments")
    
    Group.add_argument(
        "--loss_coeff", 
        type=list, 
        default=None
    ) # [0.3, 0.3, 1]
    Group.add_argument(
        "--loss_name", 
        type=str, 
        default="cross_entropy",
        help="The name of loss function",
    )
    Group.add_argument(
        "--cls_loss_name",
        type=str, 
        default=None
    )
    Group.add_argument(
        "--seg_loss_name", 
        type=str,
        default=None
    )
    Group.add_argument(
        "--det_loss_name", 
        type=str,
        default=None
    )
    Group.add_argument(
        "--insseg_loss_name", 
        type=str, 
        default=None
    )
    Group.add_argument(
        "--reg_loss_name", 
        type=str, 
        default="mse",
    )
    Group.add_argument(
        "--grad_clip", 
        type=float, 
        default=10.0,
        help="Value of gradient clip, \
            1/5 for classification and 10 for segmentation",
    )
    
    Group.add_argument(
        "--class_weights", 
        action="store_true",
        help="Use class sensitive loss?",
    )
    Group.add_argument(
        "--label_smoothing", 
        type=float, 
        default=0, # 0.1
        help="The label smoothing params for cross entropy",
    )
    Group.add_argument(
        "--loss_reduction",
        type=str, 
        default="mean",
        choices=("mean", "sum"),
        help="Choose the loss reduction method",
    )
    Group.add_argument(
        "--aux_weight",
        type=float, 
        default=0.4,
        help="The loss weight of segmentation auxiliary branch",
    )
    Group.add_argument(
        "--ignore_idx", 
        type=int, 
        default=-100,
        help="Ignore background in segmentation loss calculation",
    )
    ## multibox
    Group.add_argument(
        "--max_monitor_iter", 
        type=int, 
        default=-1,
        help="The maximum monitor iteration for multibox loss",
    )
    Group.add_argument(
        "--update_wt_freq", 
        type=int, 
        default=None,
        help="Frequency to update wegiht",
    )
    ## ntxent
    Group.add_argument(
        "--temperature", 
        type=float, 
        default=0.5,
        help="Temperature for ntxent",
    )
    ## mae
    Group.add_argument(
        "--norm_pix_loss", 
        action="store_false",
        help="The target for better representation learning",
    )
    
    return Parser


def argumentsMetrics(Parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    # metric except loss
    Group = Parser.add_argument_group(title="metric arguments")
    
    Group.add_argument(
        "--metric_indicator", 
        type=str, 
        default=None,
        choices=("loss", "top1acc", "miou", "map", "mdice"),
        help="The metric as the indicator to save best model",
    )
    Group.add_argument(
        "--metric_type", 
        type=str, 
        default="micro",
        choices=("micro", "macro"),
        help="Methods of averaging metrics",
    )
    Group.add_argument(
        "--save_metric",
        nargs="+",
        type=str,
        default=["loss"],
        help="Name of statistics",
    )
    Group.add_argument(
        "--metric_scale",
        type=float,
        default=100.,
        help="Converting metric from [0, 1] to [0, s]",
    )
    
    ## supplementary metrics for classification
    """
    Using F1-score foe evaluation on ImageNet ab CIFAR image classification (Table 1) is unclear. 
    Top-1 and top-5 accuracy are common metrics widely used in papers and public benchmarks. 
    Moreover, identical values in Accuracy and F1-score columns in Table 1 raises the question of 
    the difference between the two.
    """
    Group.add_argument(
        "--sup_metrics", 
        action="store_true",
        help="For small dataset. Supplementary metrics for classifcation, \
            including recall, precision, specificity, F1Score",
    )
    
    return Parser
    

def argumentsOptimizer(Parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    Group = Parser.add_argument_group(title="optimizer arguments")
    
    # optim
    Group.add_argument(
        "--optim", 
        default="adamw",
    )
    """
    OptimChoices = ["adam", "adamw", "adamax", "sgd", "asgd",
                "rmsprop", "rprop", "lbfgs", "adadelta", "adagrad",
                "lars"]
    """
    Group.add_argument(
        "--weight_decay",
        type=float, 
        default=0.02,
        help="Weight decay in optimiser",
    )
    Group.add_argument(
        "--milestones",
        type=int,
        default=None,
        help="Milestones for learning rate decay",
    )
    ## adam
    Group.add_argument(
        "--beta1", 
        type=float, 
        default=0.9
    )
    Group.add_argument(
        "--beta2", 
        type=float, 
        default=0.999
    )
    Group.add_argument(
        "--amsgrad",
        action="store_true",
        help="Whether to use the AMSGrad variant of this algorithm from the paper \
            `On the Convergence of Adam and Beyond`_(default: False)",
    )
    ## sgd
    Group.add_argument(
        "--nesterov",
        action="store_true",
        help="Enables Nesterov momentum (default: False)",
    )
    Group.add_argument(
        "--momentum",
        type=float, 
        default=0,
        help="Momentum factor (default: 0)",
    )
    ## lars
    Group.add_argument(
        "--eta",
        type=float, 
        default=1e-3,
        help="LARS coefficient as used in the paper (default: 1e-3)",
    )
    Group.add_argument(
        "--dampening", 
        type=float, 
        default=0,
        help="Dampening for momentum (default: 0)",
    )
    
    return Parser


def argumentsScheduler(Parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    Group = Parser.add_argument_group(title="schedular arguments")
    
    Group.add_argument(
        "--schedular",
        type=str,
        default="cosine",
        choices=(
            "cosine", "cyclic", "polynomial", 
            "steplr", "multi_step", "plateau",
        ),
        help="Chose the schedular method",
    )
    Group.add_argument(
        "--warmup_init_lr",
        type=float,
        default=None, 
        help="Warming up learning rate for schedular",
    )
    Group.add_argument(
        "--max_lr",
        type=float, 
        default=None, 
        help="Maximum learning rate for schedular",
    )
    Group.add_argument(
        "--lr", 
        type=float, 
        default=None, 
        help="The minimum learning rate",
    )
    Group.add_argument(
        "--lr_factor", 
        type=float, 
        default=6.4e-5, 
        help="The divior of divident max_lr and quotient lr, for sgd",
    )
    Group.add_argument(
        "--lr_decay",
        action="store_false",
        help="Learning rate decay",
    )
    
    return Parser


def argumentsDDP(Parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    Group = Parser.add_argument_group(title="DDP arguments")
    
    Group.add_argument(
        "--ddp", 
        action="store_true",
        help="-1 for CPU, use comma for multiple gpus",
    )
    Group.add_argument(
        "--gpus", 
        default="0", 
        help="-1 for CPU, use comma for multiple gpus",
    )
    Group.add_argument(
        "--rank",
        type=int,
        default=0,
        help="Node rank for distributed training. Defaults to 0",
    )
    Group.add_argument(
        "--world_size",
        type=int,
        default=-1,
        help="World size for DDP. Defaults to -1, meaning use all GPUs",
    )
    Group.add_argument(
        "--num_workers", 
        type=int, 
        default=None,
        help="Dataloader threads. 0 for single-thread",
    )
    Group.add_argument(
        "--pin_memory",
        action="store_false",
        help="Use pin_memory in dataloader threads.",
    )
    Group.add_argument(
        "--prefetch_factor",
        type=int,
        default=None,
        help="Number of samples loaded in advance by each data worker. Defaults to 2.",
    )

    Group.add_argument(
        "--empty_cache", 
        action="store_true",
        help="Releases all unoccupied cached memory in milestone",
    )
    
    return Parser


def argumentsCommon(Parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    Group = Parser.add_argument_group(title="common arguments")
    
    # seed
    Group.add_argument(
        "--seed",
        type=int, 
        default=999,
        help="Seed for reproduction",
    )
    
    # train
    Group.add_argument(
        "--epochs",
        type=int, 
        default=70,
        help="Total training epochs",
    )
    Group.add_argument(
        "--batch_size", 
        type=int,
        default=4,
        help="Batch size",
    ) # 8 multiple
    Group.add_argument(
        "--max_train_iters", 
        type=int, 
        default=None,
        help="The total training iterations",
    )
    Group.add_argument(
        "--exp_base", 
        type=str,
        default="exp",
        help="The base folder of exp output",
    ) 
    Group.add_argument(
        "--exp_level", 
        type=str, 
        default="",
        help="The focus comparison name for a new level of folder",
    ) 
    Group.add_argument(
        "--target_exp",
        type=int,
        default=None,
        help="The target exp folder location for supplement",
    )
    Group.add_argument(
        "--target_supplement",
        type=int, 
        default=0,
        help="Start training round, starting from 0",
    )
    Group.add_argument(
        "--save_point",
        type=str, 
        nargs="+",
        default=[None], # 30,60,90
        help="When to save the model to disk",
    )
    Group.add_argument(
        "--clsval_mode", 
        type=str, 
        default="linear",
        choices=("linear", "5nn"),
        help="Choose the classifcation accuracy computation method",
    ) # only in classification task
    Group.add_argument(
        "--cpu_5nn", 
        action="store_false",
        help="Use 5nn in CPU to save memory",
    ) # only in classification task
    Group.add_argument(
        "--knn_k",
        type=int, 
        default=5,
        help="The numer of nearest neighbor in kNN monitor",
    )
    Group.add_argument(
        "--val_start_epoch",
        type=int, 
        default=None,
        help="The validation starting point",
    )
    
    return Parser