"""Module to create Snapshot Serengeti dataset based on PyTorch ImageFolder"""

import torchvision.transforms as transforms
import torchvision.datasets as datasets

def get_split(split_name, dataset_dir, image_size):
    if split_name == 'train':
        dataset = datasets.ImageFolder(
            dataset_dir,
            transforms.Compose([
                transforms.RandomResizedCrop(image_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ]))
    elif split_name == 'val':
        dataset = datasets.ImageFolder(
            dataset_dir,
            transforms.Compose([
                transforms.Resize(image_size),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
            ]))
    else:
        raise ValueError("Split name '%s' unknown" % split_name)
    return dataset
