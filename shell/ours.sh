declare -a ModelNames=(mobilevimxs mobilevims)
ArrayLength=${#ModelNames[@]}
for (( i=0; i<${ArrayLength}; i++ ));
do
    CUDA_VISIBLE_DEVICES=0 python main.py --gpus 0 --task segmentation --setname PENGWIN --num_repeat 3 --num_workers 4 --epochs 100 --batch_size 4 --random_aug_prob 0.2 --model_name ${ModelNames[$i]} --use_mamba --mamba_block 1 --mamba_dimin 1 --mamba_dualdirection --seg_head_name fg --seg_feature_guide 3 --sup_method common --optim adamw --lr 1.6e-7 --weight_decay 0.03 --loss_name dice_ce --metric_type macro
    CUDA_VISIBLE_DEVICES=0 python main.py --gpus 0 --task segmentation --setname ATLAS_3D --num_repeat 3 --num_workers 4 --epochs 100 --batch_size 4 --random_aug_prob 0.2 --model_name ${ModelNames[$i]} --use_mamba --mamba_block 1 --mamba_dimin 1 --mamba_dualdirection --seg_head_name fg --seg_feature_guide 3 --sup_method common --optim adamw --lr 1.6e-7 --weight_decay 0.03 --loss_name dice_ce --metric_type macro
    CUDA_VISIBLE_DEVICES=0 python main.py --gpus 0 --task segmentation --setname BraTS2024T3 --num_repeat 3 --num_workers 4 --epochs 100 --batch_size 4 --random_aug_prob 0.2 --model_name ${ModelNames[$i]} --use_mamba --mamba_block 1 --mamba_dimin 1 --mamba_dualdirection --seg_head_name fg --seg_feature_guide 3 --sup_method common --optim adamw --lr 1.6e-7 --weight_decay 0.03 --loss_name dice_ce --metric_type macro
    CUDA_VISIBLE_DEVICES=0 python main.py --gpus 0 --task segmentation --setname ToothFairy2 --num_repeat 3 --num_workers 4 --epochs 100 --batch_size 4 --random_aug_prob 0.2 --model_name ${ModelNames[$i]} --use_mamba --mamba_block 1 --mamba_dimin 1 --mamba_dualdirection --seg_head_name fg --seg_feature_guide 3 --sup_method common --optim adamw --lr 1.6e-7 --weight_decay 0.03 --loss_name dice_ce --metric_type macro
done
