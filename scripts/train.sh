# This script will train a peak response mapping
# network on the Snapshot Serengeti data

python ../main.py \
    --dataset_dir=/Users/aleksandardjuric/Desktop/ss_data/dataset_512 \
    --model_dir=/Users/aleksandardjuric/Desktop/models \
    --checkpoint=/Users/aleksandardjuric/Desktop/models/checkpoint_densenet_fcn-10.pth.tar \
    --model_arch=densenet_fcn \
    --train=True \
    --evaluate=False \
    --inference=False \
    --batch_size=4 \
    --lr=0.01 \
    --image_size=448 \
    --randomize_params=classifier \
    --trainable_params=classifier \
    --start_epoch=0
