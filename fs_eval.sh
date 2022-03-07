export PYTHONPATH=./:$PYTHONPATH
dataset=cifar10

for model_dir in ../experiments/$dataset/fsat/*; do
    CUDA_VISIBLE_DEVICES=0 python3 fs_eval.py \
        --data_dir=../$dataset-data \
        --model_dir=$model_dir \
        --init_model_pass=latest \
        --attack=True \
        --attack_method_list=natural-fgsm-pgd7-pgd20-cw7-cw20 \
        --dataset=$dataset \
        --batch_size_test=64 \
        --resume
done
