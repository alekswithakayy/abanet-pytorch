# This script will analyze videos, directories or photos

python ../analyze.py \
    --model_arch=densenet_fcn \
    --checkpoint_species=/Users/aleksandardjuric/Desktop/models/checkpoint_densenet_fcn-10.pth.tar \
    --class_list_species=/Users/aleksandardjuric/Desktop/ss_data/species_list.txt \
    --checkpoint_counting=/Users/aleksandardjuric/Desktop/checkpoint_densenet_fcn-6.pth.tar\
    --class_list_counting=/Users/aleksandardjuric/Desktop/ss_data/count_list.txt \
    --results_dir=/Users/aleksandardjuric/Desktop/test/results \
    --inference_dir=/Users/aleksandardjuric/Desktop/test/videos \
    --image_size=448 \
