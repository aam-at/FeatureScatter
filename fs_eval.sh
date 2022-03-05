export PYTHONPATH=./:$PYTHONPATH

dataset=cifar10
model=preactresnet18
CUDA_VISIBLE_DEVICES=1 python3 fs_eval.py \
    --model_dir=$model_dir \
    --init_model_pass=199 \
    --attack_method_list=natural-fgsm-pgd7-pgd20-cw7-cw20 \
    --dataset=$dataset \
    --batch_size_test=64 \
    --resume
