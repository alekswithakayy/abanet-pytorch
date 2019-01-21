# This script will train a peak response mapping
# network on the Snapshot Serengeti data

python ../main.py \
    --dataset_dir=/Users/aleksandardjuric/Desktop/ss_data/dataset_512 \
    --model_dir=/Users/aleksandardjuric/Desktop/models \
    --checkpoint=/Users/aleksandardjuric/Desktop/models/model_best.pth.tar \
    --train=False \
    --evaluate=True \
    --inference=False \
    --batch_size=16 \
    --lr=0.01 \
    --peak_std=0.75 \
    --print_freq=1
