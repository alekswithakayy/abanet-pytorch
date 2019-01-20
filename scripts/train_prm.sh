# This script will train a peak response mapping
# network on the Snapshot Serengeti data

python ../main.py \
    --dataset_dir=/Users/aleksandardjuric/Desktop/ss_data/dataset \
    --model_dir=/Users/aleksandardjuric/Desktop/models \
    --checkpoint=/Users/aleksandardjuric/Desktop/models/model_best.pth.tar \
    --train=False \
    --evaluate=False \
    --inference=True \
    --inference_dir=/Users/aleksandardjuric/Desktop/test \
    --batch_size=16 \
    --lr=0.01 \
