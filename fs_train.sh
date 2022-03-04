export PYTHONPATH=./:$PYTHONPATH
dataset=cifar10
model=preactresnet18
root_dir=../experiments/$dataset/fsat2/
CUDA_VISIBLE_DEVICES=1 python3 fs_main.py \
    --adv_mode='feature_scatter' \
    --lr=0.1 \
    --root_dir=$root_dir \
    --model=$model \
    --train=standard \
    --mixup_alpha=0.0 \
    --ls-factor=0.5 \
    --random-start \
    --max_epoch=200 \
    --save_epochs=10 \
    --decay_epoch1=100 \
    --decay_epoch2=150 \
    --batch_size_train=64 \
    --dataset=$dataset \
    --job_id 0

