# declare array variables
declare -a SetNames=(ATLAS_3D PENGWIN)
ArrayLength=${#SetNames[@]}
# loop through the above array
for (( i=0; i<${ArrayLength}; i++ ));
do
    ## default settings
    CUDA_VISIBLE_DEVICES=0 python main.py --gpus 0 --task segmentation --setname ${SetNames[$i]} --num_repeat 3 --num_workers 4 --epochs 100 --batch_size 4 --random_aug_prob 0.2 --model_name mobilevims --use_mamba --mamba_block 1 --mamba_dimin 1 --mamba_dualdirection --seg_head_name fg --seg_feature_guide 3 --sup_method common --optim adamw --lr 1.6e-7 --weight_decay 0.03 --loss_name dice_ce --metric_type macro --exp_base exp_ablation
    ## mamba dimension-independent
    CUDA_VISIBLE_DEVICES=0 python main.py --gpus 0 --task segmentation --setname ${SetNames[$i]} --num_repeat 3 --num_workers 4 --epochs 100 --batch_size 4 --random_aug_prob 0.2 --model_name mobilevims --use_mamba --mamba_block 1 --mamba_dimin 0 --mamba_dualdirection --seg_head_name fg --seg_feature_guide 3 --sup_method common --optim adamw --lr 1.6e-7 --weight_decay 0.03 --loss_name dice_ce --metric_type macro --exp_base exp_ablation
    ## bidirectional
    CUDA_VISIBLE_DEVICES=0 python main.py --gpus 0 --task segmentation --setname ${SetNames[$i]} --num_repeat 3 --num_workers 4 --epochs 100 --batch_size 4 --random_aug_prob 0.2 --model_name mobilevims --use_mamba --mamba_block 1 --mamba_dimin 1 --seg_head_name fg --seg_feature_guide 3 --sup_method common --optim adamw --lr 1.6e-7 --weight_decay 0.03 --loss_name dice_ce --metric_type macro --exp_base exp_ablation
    ## fg_module
    CUDA_VISIBLE_DEVICES=0 python main.py --gpus 0 --task segmentation --setname ${SetNames[$i]} --num_repeat 3 --num_workers 4 --epochs 100 --batch_size 4 --random_aug_prob 0.2 --model_name mobilevims --use_mamba --mamba_block 1 --mamba_dimin 1 --mamba_dualdirection --seg_head_name fg --seg_feature_guide 0 --sup_method common --optim adamw --lr 1.6e-7 --weight_decay 0.03 --loss_name dice_ce --metric_type macro --exp_base exp_ablation
    ## triangle
    CUDA_VISIBLE_DEVICES=0 python main.py --gpus 0 --task segmentation --setname ${SetNames[$i]} --num_repeat 3 --num_workers 4 --epochs 100 --batch_size 4 --random_aug_prob 0.2 --model_name mobilevims --mamba_block 1 --mamba_dimin 0 --seg_head_name fg --seg_feature_guide 0 --sup_method common --optim adamw --lr 1.6e-7 --weight_decay 0.03 --loss_name dice_ce --metric_type macro --exp_base exp_ablation
    CUDA_VISIBLE_DEVICES=0 python main.py --gpus 0 --task segmentation --setname ${SetNames[$i]} --num_repeat 3 --num_workers 4 --epochs 100 --batch_size 4 --random_aug_prob 0.2 --model_name mobilevims --use_mamba --mamba_block 1 --mamba_dimin 0 --seg_head_name fg --seg_feature_guide 3 --sup_method common --optim adamw --lr 1.6e-7 --weight_decay 0.03 --loss_name dice_ce --metric_type macro --exp_base exp_ablation
done