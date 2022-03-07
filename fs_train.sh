export PYTHONPATH=./:$PYTHONPATH
dataset=cifar10
model_dir=../experiments/$dataset/model_0
mkdir -p $model_dir
CUDA_VISIBLE_DEVICES=0 python3 fs_main.py \
    --adv_mode='feature_scatter' \
    --lr=0.1 \
    --model_dir=$model_dir \
    --init_model_pass=latest \
    --ls_factor=0.5 \
    --random_start \
    --max_epoch=200 \
    --save_epochs=100 \
    --decay_epoch1=100 \
    --decay_epoch2=150 \
    --batch_size_train=64 \
    --dataset=$dataset

