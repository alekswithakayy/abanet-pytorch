# This script will download and build Snapshot Serengeti data

python ./download_snapshot_serengeti.py \
    /home/ss_data \
    --num_threads=16

python ./build_snapshot_serengeti.py \
    /home/ss_data \
    /home/ss_data/dataset \
    --image_size=256 \
    --num_threads=16
