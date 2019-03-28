"Returns split for a pytorch image folder dataset"

import torchvision.transforms as transforms
import torchvision.datasets as datasets

def get_split(split_name, dataset_dir, args):
    if not args.image_size:
        raise ValueError('No image size given!')
    image_size = args.image_size
    if split_name == 'train':
        dataset = datasets.ImageFolder(
            dataset_dir,
            transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ]))
    elif split_name == 'test':
        dataset = datasets.ImageFolder(
            dataset_dir,
            transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
            ]))
    else:
        raise ValueError("Split name '%s' unknown" % split_name)
    return dataset
