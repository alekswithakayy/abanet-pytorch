# This script will analyze videos, directories or photos

python ../analyze.py \
    --model_arch=densenet_fcn \
    --checkpoint_species=/Users/aleksandardjuric/Desktop/checkpoint_densenet_fcn-4-species.pth.tar \
    --class_list_species=/Users/aleksandardjuric/Desktop/ss_data/species_list.txt \
    --checkpoint_counting=/Users/aleksandardjuric/Desktop/checkpoint_densenet_fcn-5-count.pth.tar \
    --class_list_counting=/Users/aleksandardjuric/Desktop/ss_data/count_list.txt \
    --results_dir=/Users/aleksandardjuric/Desktop/test/results_photos \
    --inference_dir=/Users/aleksandardjuric/Desktop/test/photos \
    --image_size=224 \
