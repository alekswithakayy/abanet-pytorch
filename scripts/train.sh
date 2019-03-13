# This script will train a peak response mapping
# network on the Snapshot Serengeti data

python ../train.py \
    --model_arch=densenet_fcn \
    --dataset_name=snapshot_serengeti \
    --dataset_dir=/Users/aleksandardjuric/Desktop/samples \
    --models_dir=/Main/projects/animal_behaviour_analysis/models \
    --checkpoint_file=/Main/projects/animal_behaviour_analysis/models/checkpoint_densenet_fcn-10.pth.tar \
    --pretrained=False \
    --params_to_train=classifier \
    --params_to_randomize=classifier \
    --train=True \
    --validate=False \
    --epochs=3 \
    --start_epoch=0 \
    --batch_size=4 \
    --criterion=BCEWithLogitsLoss \
    --optimizer=SGD \
    --num_threads=4 \
    --image_size=330 \
    --lr=0.01 \
    --lr_decay=0.1 \
    --lr_decay_epochs=1 \
    --momentum=0.9 \
    --weight_decay=1e-4 \
    --print_freq=10
