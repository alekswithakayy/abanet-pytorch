"""Module to create Snapshot Serengeti dataset based on PyTorch ImageFolder"""

import torchvision.transforms as transforms
import torchvision.datasets as datasets

def get_split(split_name, dataset_dir):
    if split_name == 'train':
        dataset = datasets.ImageFolder(
            dataset_dir,
            transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ]))
    elif split_name == 'val':
        dataset = datasets.ImageFolder(
            dataset_dir,
            transforms.Compose([
                transforms.Resize(224),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
            ]))
    else:
        raise ValueError("Split name '%s' unknown" % split_name)
    return dataset
