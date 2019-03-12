# This script will train a peak response mapping
# network on the Snapshot Serengeti data

python ../train.py \
    --model_arch=densenet_fcn \
    --dataset_name=snapshot_serengeti \
    --dataset_dir=/Main/projects/animal_behaviour_analysis/ss_data/dataset_512 \
    --models_dir=/Main/projects/animal_behaviour_analysis/models \
    --checkpoint_file=/Main/projects/animal_behaviour_analysis/models/checkpoint_densenet_fcn-10.pth.tar \
    --train=True \
    --validate=False \
    --batch_size=4 \
    --lr=0.01 \
    --image_size=448 \
    --params_to_randomize=classifier \
    --params_to_train=classifier \
    --start_epoch=0
