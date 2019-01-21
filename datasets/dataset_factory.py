"""A factory-pattern class which returns PyTorch datasets."""

from datasets.snapshot_serengeti import snapshot_serengeti

datasets_map = {
    'snapshot_serengeti': snapshot_serengeti,
}

def get_dataset(name, split_name, dataset_dir, image_size):
    """Given a dataset name and a split name returns a PyTorch dataset."""
    if name not in datasets_map:
        raise ValueError('Name of dataset unknown %s' % name)
    return datasets_map[name].get_split(split_name, dataset_dir, image_size)
