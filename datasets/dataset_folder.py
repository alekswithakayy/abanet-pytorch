"Returns split for a pytorch dataset folder dataset"

import torchvision.transforms as transforms
import torchvision.datasets as datasets

from PIL import Image

IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp')

def pil_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')

def get_split(split_name, dataset_dir, args):
    if not args.image_size:
        raise ValueError('No image size given!')
    if split_name == 'train':
        dataset = datasets.DatasetFolder(
            dataset_dir,
            pil_loader,
            IMG_EXTENSIONS,
            transform=transforms.Compose([
                transforms.Resize(args.image_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ]))
    elif split_name == 'test':
        dataset = datasets.DatasetFolder(
            dataset_dir,
            pil_loader,
            IMG_EXTENSIONS,
            transform=transforms.Compose([
                transforms.Resize(args.image_size),
                transforms.ToTensor(),
            ]))
    else:
        raise ValueError("Split name '%s' unknown" % split_name)
    return dataset
