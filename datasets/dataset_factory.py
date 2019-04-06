"""Given a dataset name and a split name returns a PyTorch dataset."""

import os

from datasets import dataset_folder
from datasets.snapshot_serengeti import snapshot_serengeti

datasets_map = {
    'image_folder': dataset_folder,
    'snapshot_serengeti': snapshot_serengeti,
}

def get_dataset(split_name, args):
    dataset_dir = os.path.join(args.dataset_dir, split_name)
    if os.path.isdir(dataset_dir):
        if args.dataset not in datasets_map:
            raise ValueError("Name of dataset '%s' unknown" % args.dataset)
        return datasets_map[args.dataset].get_split(split_name, dataset_dir, args)
    else:
        raise ValueError("%s does not exist" % dataset_dir)
