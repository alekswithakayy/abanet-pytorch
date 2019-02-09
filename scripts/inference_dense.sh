# This script will train a peak response mapping
# network on the Snapshot Serengeti data

python ../main.py \
    --dataset_dir=/Users/aleksandardjuric/Desktop/ss_data/dataset_512 \
    --model_dir=/Users/aleksandardjuric/Desktop/models \
    --checkpoint=/Users/aleksandardjuric/Desktop/checkpoint_densenet_fcn-10.pth.tar \
    --model_arch=densenet_fcn \
    --train=False \
    --evaluate=False \
    --inference=True \
    --inference_dir=/Users/aleksandardjuric/Desktop/ \
    --batch_size=16 \
    --lr=0.01 \
    --image_size=448 \
    --prm=False \
